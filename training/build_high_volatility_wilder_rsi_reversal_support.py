"""Build outcome-blind source support for frozen HVWRSI-24."""
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

from training import preregister_high_volatility_wilder_rsi_reversal as prereg

ENV_FILE = "/home/pakchu/rllm/.env"
START = pd.Timestamp("2020-01-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA = "4a2a6e33c0cfde58e58e145ed1d64fb061c2ad6a609a7d89175af14a92b7aa1a"
REG = prereg.build()
P = REG["policy"]
SPLITS = {name: tuple(map(pd.Timestamp, window)) for name, window in REG["stages"].items()}
GATES = REG["source_support_gates"]
CONTROLS = tuple(REG["diagnostic_controls"]["names"])
QUERY = """SELECT ts,open,high,low,close FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts"""
ROOT = Path("data/high_volatility_wilder_rsi_reversal_sources_2023_2026")
PANEL = ROOT / "daily_states.csv.gz"
MANIFEST = ROOT / "manifest.json"
CLOCK = Path("data/high_volatility_wilder_rsi_reversal_clocks_2023_2026.csv.gz")
SPLIT_DIR = Path("data/high_volatility_wilder_rsi_reversal_split_clocks_2023_2026")
CONTROL_DIR = Path("data/high_volatility_wilder_rsi_reversal_controls_2023_2026")
RESULT = Path("results/high_volatility_wilder_rsi_reversal_support_2026-08-11.json")
BUILDER = Path(__file__).relative_to(Path.cwd())
PANEL_COLUMNS = ("source_day","feature_available_time","decision_time","source_valid","daily_close","daily_return","avg_gain","avg_loss","rsi","variation","variation_rank","eligible")
CLOCK_COLUMNS = ("candidate","control","split","source_day","feature_available_time","decision_time","entry_time","exit_time","side","daily_close","daily_return","avg_gain","avg_loss","rsi","variation","variation_rank","eligible")



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
        prior = np.asarray(history[-P["variation_history_days"] :], float)
        if math.isfinite(value) and len(prior) >= P["minimum_variation_history_days"]:
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
        raise RuntimeError("HVWRSI source schema drift")
    source = frame.copy()
    source["ts"] = pd.to_datetime(source.ts, utc=True, errors="coerce")
    for column in ("open", "high", "low", "close"):
        source[column] = pd.to_numeric(source[column], errors="coerce")
    if source.ts.isna().any() or source.ts.duplicated().any():
        raise RuntimeError("HVWRSI invalid source key")
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


def wilder_rsi(changes: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    values=pd.to_numeric(changes,errors="coerce").to_numpy(float);n=len(values);gain=np.full(n,np.nan);loss=np.full(n,np.nan);rsi=np.full(n,np.nan);period=P["rsi_days"]
    if n<=period:return pd.Series(gain,index=changes.index),pd.Series(loss,index=changes.index),pd.Series(rsi,index=changes.index)
    for i in range(period,n):
        if i==period:
            window=values[1:period+1]
            if not np.isfinite(window).all():continue
            g=float(np.maximum(window,0).mean());l=float(np.maximum(-window,0).mean())
        else:
            if not math.isfinite(gain[i-1]) or not math.isfinite(loss[i-1]) or not math.isfinite(values[i]):continue
            g=((period-1)*gain[i-1]+max(values[i],0))/period;l=((period-1)*loss[i-1]+max(-values[i],0))/period
        gain[i]=g;loss[i]=l
        if g==0 and l==0:continue
        rsi[i]=100. if l==0 else 0. if g==0 else 100-100/(1+g/l)
    return pd.Series(gain,index=changes.index),pd.Series(loss,index=changes.index),pd.Series(rsi,index=changes.index)

def build_panel(raw: pd.DataFrame) -> pd.DataFrame:
    source=prepare(raw).reindex(pd.date_range(START,END,freq="1min",inclusive="left"));groups=source.groupby(source.index.floor("1d"),sort=True)
    daily=pd.DataFrame({"rows":groups.row_valid.sum(),"daily_close":groups.close.last()});daily["valid_day"]=daily.rows.eq(1440)&np.isfinite(daily.daily_close)&daily.daily_close.gt(0)
    daily["daily_return"]=np.log(daily.daily_close/daily.daily_close.shift(1)).where(daily.valid_day&daily.valid_day.shift(1,fill_value=False))
    changes=(daily.daily_close-daily.daily_close.shift(1)).where(daily.valid_day&daily.valid_day.shift(1,fill_value=False));daily["avg_gain"],daily["avg_loss"],daily["rsi"]=wilder_rsi(changes)
    daily["variation"]=np.sqrt(np.square(daily.daily_return).rolling(P["variation_days"],min_periods=P["variation_days"]).sum())
    daily["source_valid"]=daily.valid_day&np.isfinite(daily[["daily_return","avg_gain","avg_loss","rsi","variation"]]).all(axis=1)&daily.variation.gt(0)
    panel=daily.reset_index(names="source_day");panel["feature_available_time"]=panel.source_day+pd.Timedelta("1d");panel["decision_time"]=panel.feature_available_time
    panel["variation_rank"]=prior_rank(panel.variation.where(panel.source_valid));panel["eligible"]=panel.source_valid&(panel.rsi.le(P["oversold_max"])|panel.rsi.ge(P["overbought_min"]))&panel.variation_rank.ge(P["variation_rank_min"])
    return panel.loc[:,PANEL_COLUMNS]


def active(panel: pd.DataFrame, control: str = "primary"):
    if control not in ("primary",*CONTROLS):raise ValueError(control)
    used=panel.copy();valid=used.source_valid.eq(True);rsi=pd.to_numeric(used.rsi,errors="coerce");extreme=rsi.le(P["oversold_max"])|rsi.ge(P["overbought_min"]);variation=used.variation_rank.ge(P["variation_rank_min"]);side=pd.Series(np.where(rsi.le(P["oversold_max"]),1,np.where(rsi.ge(P["overbought_min"]),-1,0)),index=used.index,dtype=int);state=valid&side.ne(0)&extreme&variation
    if control=="no_variation_gate":state=valid&side.ne(0)&extreme
    elif control=="one_day_stale_rsi":
        stale=valid.shift(1,fill_value=False);side=side.shift(1,fill_value=0);state=stale&side.ne(0)&extreme.shift(1,fill_value=False)&variation.shift(1,fill_value=False)
    if control=="direction_flip":side=-side
    if control=="forced_long":side=pd.Series(1,index=side.index,dtype=int)
    return state&side.ne(0),side,used


def build_clock(panel: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    activity, side, used = active(panel, control)
    rows = []
    for index in panel.index[activity]:
        day = pd.Timestamp(panel.at[index, "source_day"])
        decision = pd.Timestamp(panel.at[index, "decision_time"])
        entry = decision + pd.Timedelta("5min")
        exit_time = entry + pd.Timedelta("24h")
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
        raise RuntimeError("HVWRSI prereg drift")
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
        "protocol_version": "hvwrsi_24_source_v1",
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
        "protocol_version": "hvwrsi_24_source_support_v1",
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
