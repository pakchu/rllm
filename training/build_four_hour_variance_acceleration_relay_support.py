"""Materialize source-only FHVAR-2 clocks."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_four_hour_variance_acceleration_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
START = pd.Timestamp("2023-06-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
SOURCE_DIR = Path("data/four_hour_variance_acceleration_relay_sources_2023_2026")
PANEL = SOURCE_DIR / "four_hour_variance_panel.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/four_hour_variance_acceleration_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/four_hour_variance_acceleration_relay_controls_2023_2026")
RESULT = Path("results/four_hour_variance_acceleration_relay_support_2026-08-08.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_volatility_gate",
    "no_acceleration_gate",
    "one_boundary_stale_geometry",
    "direction_fade",
)
COLUMNS = (
    "candidate",
    "control",
    "split",
    "decision_time",
    "feature_available_time",
    "entry_time",
    "exit_time",
    "side",
    "first_open",
    "first_close",
    "second_open",
    "second_close",
    "first_return",
    "second_return",
    "first_variation",
    "second_variation",
    "full_variation",
    "variation_rank",
    "second_to_first_variation",
)
QUERY = """
SELECT ts,open,high,low,close
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts
"""


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def strict_prior_midrank(
    values: pd.Series, lookback: int = 180, minimum: int = 120
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


def _variance_panel(bars: pd.DataFrame) -> pd.DataFrame:
    bars = bars.copy()
    bars["ts"] = pd.to_datetime(bars["ts"], utc=True)
    for column in ("open", "high", "low", "close"):
        bars[column] = pd.to_numeric(bars[column], errors="coerce")
    bars = bars.drop_duplicates("ts", keep=False).set_index("ts").sort_index()
    decisions = pd.date_range(START.ceil("4h"), END, freq="4h", inclusive="left")
    rows = []
    for decision in decisions:
        expected = pd.date_range(
            decision - pd.Timedelta(hours=4), decision, freq="1min", inclusive="left"
        )
        window = bars.reindex(expected)
        finite = np.isfinite(window[["open", "high", "low", "close"]]).all(axis=1)
        positive = window[["open", "high", "low", "close"]].gt(0).all(axis=1)
        coherent = (
            window["high"].ge(window[["open", "close"]].max(axis=1))
            & window["low"].le(window[["open", "close"]].min(axis=1))
            & window["high"].ge(window["low"])
        )
        valid = len(window) == 240 and bool((finite & positive & coherent).all())
        if valid:
            first = window.iloc[:120]
            second = window.iloc[120:]
            first_open = float(first["open"].iloc[0])
            first_close = float(first["close"].iloc[-1])
            second_open = float(second["open"].iloc[0])
            second_close = float(second["close"].iloc[-1])
            first_log_close = np.log(first["close"].astype(float))
            second_log_close = np.log(second["close"].astype(float))
            first_variation = float(first_log_close.diff().dropna().pow(2).sum())
            second_variation = float(second_log_close.diff().dropna().pow(2).sum())
            first_return = float(np.log(first_close / first_open))
            second_return = float(np.log(second_close / second_open))
        else:
            first_open = first_close = second_open = second_close = float("nan")
            first_return = second_return = float("nan")
            first_variation = second_variation = float("nan")
        rows.append(
            {
                "decision_time": decision,
                "source_rows": int(window.notna().all(axis=1).sum()),
                "source_valid": valid,
                "first_open": first_open,
                "first_close": first_close,
                "second_open": second_open,
                "second_close": second_close,
                "first_return": first_return,
                "second_return": second_return,
                "first_variation": first_variation,
                "second_variation": second_variation,
            }
        )
    panel = pd.DataFrame(rows)
    panel["full_variation"] = (panel.first_variation + panel.second_variation).where(panel.source_valid)
    panel["variation_rank"] = strict_prior_midrank(panel.full_variation)
    panel["second_to_first_variation"] = (panel.second_variation / panel.first_variation).where(
        panel.source_valid & panel.first_variation.gt(0)
    )
    return panel


def materialize() -> dict[str, Any]:
    from sqlalchemy import text

    database = postgres_engine()
    with database.connect() as connection:
        bars = pd.read_sql_query(
            text(QUERY),
            connection,
            params={"start": START.to_pydatetime(), "end": END.to_pydatetime()},
        )
    database.dispose()
    panel = _variance_panel(bars)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(panel, PANEL)
    core = {
        "protocol_version": "fhvar_2_btc_source_v1",
        "query": QUERY,
        "table": "bars_binance",
        "symbol": "BTCUSDT",
        "interval": "1m",
        "window": [START.isoformat(), END.isoformat()],
        "exact_candidate_outcomes_opened": False,
        "candidate_incidence_opened": False,
        "no_imputation": True,
        "output": {
            "path": str(PANEL),
            "sha256": sha(PANEL),
            "rows": len(panel),
            "valid_rows": int(panel.source_valid.sum()),
        },
    }
    manifest = {**core, "manifest_hash": canonical_hash(core)}
    SOURCE_MANIFEST.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def features() -> pd.DataFrame:
    frame = pd.read_csv(PANEL, compression="gzip")
    frame["decision_time"] = pd.to_datetime(frame.decision_time, utc=True)
    frame["source_valid"] = frame.source_valid.astype(str).str.lower().eq("true")
    numeric = (
        "first_open",
        "first_close",
        "second_open",
        "second_close",
        "first_return",
        "second_return",
        "first_variation",
        "second_variation",
        "full_variation",
        "variation_rank",
        "second_to_first_variation",
    )
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["signal_valid"] = (
        frame.source_valid
        & np.isfinite(frame[list(numeric)]).all(axis=1)
        & frame.full_variation.gt(0)
    )
    return frame


def conditions(frame: pd.DataFrame, control: str) -> tuple[pd.Series, pd.Series]:
    ratio = frame.second_to_first_variation
    first_return = frame.first_return
    second_return = frame.second_return
    rank = frame.variation_rank
    if control == "one_boundary_stale_geometry":
        ratio = ratio.shift(1)
        first_return = first_return.shift(1)
        second_return = second_return.shift(1)
        rank = rank.shift(1)
    volatility_gate = (
        pd.Series(True, index=frame.index)
        if control == "no_volatility_gate"
        else rank.ge(0.65)
    )
    acceleration_gate = (
        pd.Series(True, index=frame.index)
        if control == "no_acceleration_gate"
        else ratio.ge(1.5)
    )
    long = first_return.gt(0) & second_return.gt(0)
    short = first_return.lt(0) & second_return.lt(0)
    active = (
        frame.signal_valid
        & np.isfinite(ratio)
        & np.isfinite(first_return)
        & np.isfinite(second_return)
        & np.isfinite(rank)
        & volatility_gate
        & acceleration_gate
        & (long | short)
    )
    side = pd.Series(np.where(long, 1, -1), index=frame.index)
    if control == "direction_fade":
        side = -side
    return active, side


def clock(frame: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    active, side = conditions(frame, control)
    rows = []
    for index in frame.index[active]:
        decision = pd.Timestamp(frame.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=5)
        exit_ = entry + pd.Timedelta(hours=2)
        split = next(
            (
                name
                for name, (start, end) in SPLITS.items()
                if entry >= start and exit_ <= end
            ),
            None,
        )
        if split is None:
            continue
        rows.append(
            {
                "candidate": "FHVAR-2",
                "control": control,
                "split": split,
                "decision_time": decision,
                "feature_available_time": decision,
                "entry_time": entry,
                "exit_time": exit_,
                "side": int(side.at[index]),
                "first_open": float(frame.at[index, "first_open"]),
                "first_close": float(frame.at[index, "first_close"]),
                "second_open": float(frame.at[index, "second_open"]),
                "second_close": float(frame.at[index, "second_close"]),
                "first_return": float(frame.at[index, "first_return"]),
                "second_return": float(frame.at[index, "second_return"]),
                "first_variation": float(frame.at[index, "first_variation"]),
                "second_variation": float(frame.at[index, "second_variation"]),
                "full_variation": float(frame.at[index, "full_variation"]),
                "variation_rank": float(frame.at[index, "variation_rank"]),
                "second_to_first_variation": float(frame.at[index, "second_to_first_variation"]),
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def stats(candidate: pd.DataFrame, split: str) -> dict[str, float | int]:
    subset = candidate[candidate.split.eq(split)]
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
    months = subset.entry_time.dt.strftime("%Y-%m").value_counts()
    return {
        "events": len(subset),
        "longs": longs,
        "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(subset),
        "max_month_share": int(months.max()) / len(subset),
    }


def run() -> dict[str, Any]:
    source_manifest = materialize()
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
        checks[f"{name}_side_balance"] = values["minority_side_share"] >= 0.2
        checks[f"{name}_month_concentration"] = values["max_month_share"] <= 0.45
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    passed = all(checks.values())
    core = {
        "protocol_version": "fhvar_2_source_support_v1",
        "policy_id": "FHVAR-2",
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT),
            "sha256": sha(prereg.DEFAULT_OUTPUT),
            "manifest_hash": registration["manifest_hash"],
        },
        "source_manifest": {
            "path": str(SOURCE_MANIFEST),
            "sha256": sha(SOURCE_MANIFEST),
            "manifest_hash": source_manifest["manifest_hash"],
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
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    output = run()
    print(json.dumps({"passed": output["support_passed"], "support": output["support"]}, indent=2))
