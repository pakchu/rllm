"""Build outcome-blind source support for frozen HVIHR-1."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_intraday_hour_reversal as prereg

ENV_FILE = "/home/pakchu/rllm/.env"
START = pd.Timestamp("2023-02-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA = "f47272927574f87db59b774f1aeef5c56ea3e42a52b5e6a59ab5911d064f484c"
REG = prereg.build()
P = REG["policy"]
SPLITS = {name: tuple(map(pd.Timestamp, window)) for name, window in REG["stages"].items()}
GATES = REG["source_support_gates"]
CONTROLS = tuple(REG["diagnostic_controls"]["names"])
QUERY = """SELECT ts,open,high,low,close FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts"""
ROOT = Path("data/high_volatility_intraday_hour_reversal_sources_2023_2026")
PANEL = ROOT / "daily_states.csv.gz"
MANIFEST = ROOT / "manifest.json"
CLOCK = Path("data/high_volatility_intraday_hour_reversal_clocks_2023_2026.csv.gz")
SPLIT_DIR = Path("data/high_volatility_intraday_hour_reversal_split_clocks_2023_2026")
CONTROL_DIR = Path("data/high_volatility_intraday_hour_reversal_controls_2023_2026")
RESULT = Path("results/high_volatility_intraday_hour_reversal_support_2026-08-11.json")
BUILDER = Path(__file__).relative_to(Path.cwd())
PANEL_COLUMNS = (
    "source_day", "feature_available_time", "source_valid", "predictor_return",
    "predictor_magnitude", "predictor_magnitude_rank", "realized_variation",
    "variation_rank", "eligible",
)
CLOCK_COLUMNS = (
    "candidate", "control", "split", "source_day", "feature_available_time",
    "decision_time", "entry_time", "exit_time", "side", "predictor_return",
    "predictor_magnitude", "predictor_magnitude_rank", "realized_variation",
    "variation_rank", "eligible",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def prior_rank(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(float)
    output = np.full(len(values), np.nan)
    history: list[float] = []
    for index, value in enumerate(values):
        prior = np.asarray(history[-P["history_days"] :], float)
        if math.isfinite(value) and len(prior) >= P["minimum_history_days"]:
            output[index] = (np.sum(prior < value) + 0.5 * np.sum(prior == value)) / len(prior)
        if math.isfinite(value):
            history.append(float(value))
    return pd.Series(output, index=series.index)


def postgres_engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def load_source() -> pd.DataFrame:
    from sqlalchemy import text

    database = postgres_engine()
    try:
        with database.connect() as connection:
            return pd.read_sql_query(
                text(QUERY),
                connection,
                params={"start": START.to_pydatetime(), "end": END.to_pydatetime()},
            )
    finally:
        database.dispose()


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.columns.tolist() != ["ts", "open", "high", "low", "close"]:
        raise RuntimeError("HVIHR source schema drift")
    source = frame.copy()
    source["ts"] = pd.to_datetime(source.ts, utc=True, errors="coerce")
    for column in ("open", "high", "low", "close"):
        source[column] = pd.to_numeric(source[column], errors="coerce")
    if source.ts.isna().any() or source.ts.duplicated().any():
        raise RuntimeError("HVIHR invalid source key")
    prices = source[["open", "high", "low", "close"]]
    source["row_valid"] = (
        np.isfinite(prices).all(axis=1)
        & prices.gt(0).all(axis=1)
        & source.high.ge(prices[["open", "close"]].max(axis=1))
        & source.low.le(prices[["open", "close"]].min(axis=1))
        & source.high.ge(source.low)
    )
    source["minute_return"] = np.log(source.close / source.open).where(source.row_valid)
    return source.set_index("ts").sort_index()


def build_panel(raw: pd.DataFrame) -> pd.DataFrame:
    source = prepare(raw).reindex(pd.date_range(START, END, freq="1min", inclusive="left"))
    rows: list[dict[str, Any]] = []
    for day in pd.date_range(START.normalize() + pd.Timedelta("1d"), END.normalize(), freq="1d", inclusive="left"):
        feature_time = day + pd.Timedelta("3h")
        predictor = source.loc[day + pd.Timedelta("2h") : feature_time - pd.Timedelta("1min")]
        variation = source.loc[feature_time - pd.Timedelta("1d") : feature_time - pd.Timedelta("1min")]
        valid = (
            len(predictor) == 60
            and len(variation) == 1440
            and predictor.row_valid.eq(True).all()
            and variation.row_valid.eq(True).all()
        )
        predictor_return = (
            float(np.log(predictor.close.iloc[-1] / predictor.open.iloc[0])) if valid else math.nan
        )
        realized_variation = (
            float(np.sqrt(np.square(variation.minute_return.to_numpy(float)).sum())) if valid else math.nan
        )
        valid = bool(
            valid
            and math.isfinite(predictor_return)
            and predictor_return != 0
            and math.isfinite(realized_variation)
            and realized_variation > 0
        )
        rows.append(
            {
                "source_day": day,
                "feature_available_time": feature_time,
                "source_valid": valid,
                "predictor_return": predictor_return,
                "realized_variation": realized_variation,
            }
        )
    panel = pd.DataFrame(rows)
    panel["predictor_magnitude"] = panel.predictor_return.abs()
    panel["predictor_magnitude_rank"] = prior_rank(panel.predictor_magnitude.where(panel.source_valid))
    panel["variation_rank"] = prior_rank(panel.realized_variation.where(panel.source_valid))
    panel["eligible"] = (
        panel.source_valid
        & panel.predictor_magnitude_rank.ge(P["predictor_magnitude_rank_min"])
        & panel.variation_rank.ge(P["variation_rank_min"])
    )
    return panel.loc[:, PANEL_COLUMNS]


def active(panel: pd.DataFrame, control: str = "primary"):
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    used = panel.copy()
    valid = used.source_valid.eq(True)
    magnitude = used.predictor_magnitude_rank.ge(P["predictor_magnitude_rank_min"])
    variation = used.variation_rank.ge(P["variation_rank_min"])
    side = -np.sign(pd.to_numeric(used.predictor_return, errors="coerce")).fillna(0).astype(int)
    state = valid & side.ne(0) & magnitude & variation
    if control == "no_predictor_magnitude_gate":
        state = valid & side.ne(0) & variation
    elif control == "no_variation_gate":
        state = valid & side.ne(0) & magnitude
    elif control == "one_day_stale_predictor":
        stale_valid = valid.shift(1, fill_value=False)
        side = side.shift(1, fill_value=0)
        state = stale_valid & side.ne(0) & magnitude.shift(1, fill_value=False) & variation.shift(1, fill_value=False)
    if control == "direction_flip":
        side = -side
    if control == "forced_long":
        side = pd.Series(1, index=side.index, dtype=int)
    return state & side.ne(0), side, used


def build_clock(panel: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    activity, side, used = active(panel, control)
    rows = []
    for index in panel.index[activity]:
        day = pd.Timestamp(panel.at[index, "source_day"])
        decision = day + pd.Timedelta("18h")
        entry = decision + pd.Timedelta("5min")
        exit_time = day + pd.Timedelta("19h")
        split = next(
            (name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end),
            None,
        )
        if split is None:
            continue
        rows.append(
            {
                "candidate": prereg.POLICY_ID,
                "control": control,
                "split": split,
                "source_day": day,
                "feature_available_time": pd.Timestamp(used.at[index, "feature_available_time"]),
                "decision_time": decision,
                "entry_time": entry,
                "exit_time": exit_time,
                "side": int(side.at[index]),
                **{
                    column: bool(used.at[index, column]) if column == "eligible" else float(used.at[index, column])
                    for column in CLOCK_COLUMNS[9:]
                },
            }
        )
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, float | int]:
    selected = clock[clock.split.eq(split)]
    if selected.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(selected.side.eq(1).sum())
    shorts = int(selected.side.eq(-1).sum())
    months = pd.to_datetime(selected.entry_time, utc=True).dt.strftime("%Y-%m").value_counts()
    return {
        "events": len(selected),
        "longs": longs,
        "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(selected),
        "max_month_share": int(months.max()) / len(selected),
    }


def csv_gz(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    raw = frame.to_csv(index=False, float_format="%.12g", lineterminator="\n").encode()
    with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=6, mtime=0) as output:
        output.write(raw)
    return buffer.getvalue()


def immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != content:
        raise RuntimeError(f"refusing overwrite {path}")
    path.write_bytes(content)


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVIHR prereg drift")
    raw = load_source()
    panel = build_panel(raw)
    primary = build_clock(panel)
    controls = {name: build_clock(panel, name) for name in CONTROLS}
    splits = {name: primary[primary.split.eq(name)].copy() for name in SPLITS}
    immutable(PANEL, csv_gz(panel))
    immutable(CLOCK, csv_gz(primary))
    for name, frame in controls.items():
        immutable(CONTROL_DIR / f"{name}.csv.gz", csv_gz(frame))
    for name, frame in splits.items():
        immutable(SPLIT_DIR / f"{name}.csv.gz", csv_gz(frame))
    source_core = {
        "protocol_version": "hvihr_1_source_v1",
        "query": QUERY,
        "query_sha256": hashlib.sha256(QUERY.encode()).hexdigest(),
        "window": [START.isoformat(), END.isoformat()],
        "physical_rows": len(raw),
        "builder": {"path": str(BUILDER), "sha256": sha(BUILDER)},
        "panel": {"path": str(PANEL), "sha256": sha(PANEL), "rows": len(panel), "valid_rows": int(panel.source_valid.sum())},
        "outcomes_opened": False,
        "gross9_rows_opened": False,
        "no_imputation": True,
    }
    manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}
    immutable(MANIFEST, json_bytes(manifest))
    support = {name: stats(primary, name) for name in SPLITS}
    checks = {
        key: passed
        for name, values in support.items()
        for key, passed in (
            (f"{name}_minimum_events", values["events"] >= GATES["minimum_events"][name]),
            (f"{name}_side_balance", values["minority_side_share"] >= GATES["minority_side_share_min"]),
            (f"{name}_month_concentration", values["max_month_share"] <= GATES["max_month_share"]),
        )
    }
    passed = all(checks.values())
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    core = {
        "protocol_version": "hvihr_1_source_support_v1",
        "policy_id": prereg.POLICY_ID,
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(MANIFEST), "sha256": sha(MANIFEST), "manifest_hash": manifest["manifest_hash"]},
        "completed_preentry_sources_opened": True,
        "candidate_incidence_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "funding_values_opened": False,
        "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "split_artifacts": {name: {"path": str(SPLIT_DIR / f"{name}.csv.gz"), "sha256": sha(SPLIT_DIR / f"{name}.csv.gz"), "rows": len(frame)} for name, frame in splits.items()},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(frame), "promotion_authorized": False} for name, frame in controls.items()},
        "support": support,
        "support_checks": checks,
        "support_passed": passed,
        "advance_to_gross9_novelty": passed,
        "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    immutable(RESULT, json_bytes(result))
    return result


if __name__ == "__main__":
    print(json.dumps({"passed": run()["support_passed"], "result": str(RESULT)}))
