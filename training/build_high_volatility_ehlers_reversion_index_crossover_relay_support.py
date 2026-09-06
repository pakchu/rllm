"""Build outcome-blind source support for frozen HVERI-24."""
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

from training import preregister_high_volatility_ehlers_reversion_index_crossover_relay as prereg

ENV_FILE = "/home/pakchu/rllm/.env"
START = pd.Timestamp("2020-01-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA = "23a5e28dce104959eaa5d1833002ab1c7cf307fbaa3e785335881a734b5c9c80"
REG = prereg.build()
P = REG["policy"]
SPLITS = {name: tuple(map(pd.Timestamp, window)) for name, window in REG["stages"].items()}
GATES = REG["source_support_gates"]
CONTROLS = tuple(REG["diagnostic_controls"]["names"])
QUERY = """SELECT ts,open,high,low,close FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts"""
ROOT = Path("data/high_volatility_ehlers_reversion_index_crossover_relay_sources_2023_2026")
PANEL = ROOT / "four_hour_states.csv.gz"
MANIFEST = ROOT / "manifest.json"
CLOCK = Path("data/high_volatility_ehlers_reversion_index_crossover_relay_clocks_2023_2026.csv.gz")
SPLIT_DIR = Path("data/high_volatility_ehlers_reversion_index_crossover_relay_split_clocks_2023_2026")
CONTROL_DIR = Path("data/high_volatility_ehlers_reversion_index_crossover_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_ehlers_reversion_index_crossover_relay_support_2026-08-11.json")
BUILDER = Path(__file__).relative_to(Path.cwd())
PANEL_COLUMNS = (
    "source_start", "feature_available_time", "source_valid", "bar_open", "bar_high",
    "bar_low", "bar_close", "run_length", "delta_sum", "absolute_delta_sum", "ratio",
    "smooth", "trigger", "spread", "entry_side", "ratio_cross_side", "variation", "variation_rank", "eligible",
)
CLOCK_COLUMNS = (
    "candidate", "control", "split", "source_start", "feature_available_time",
    "entry_time", "exit_time", "side", "run_length", "delta_sum", "absolute_delta_sum", "ratio",
    "smooth", "trigger", "spread", "entry_side", "ratio_cross_side", "variation", "variation_rank", "eligible",
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
        prior = np.asarray(history[-P["variation_history_decisions"] :], float)
        if math.isfinite(value) and len(prior) >= P["minimum_variation_history_decisions"]:
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
                text(QUERY), connection,
                params={"start": START.to_pydatetime(), "end": END.to_pydatetime()},
            )
    finally:
        database.dispose()


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.columns.tolist() != ["ts", "open", "high", "low", "close"]:
        raise RuntimeError("HVERI source schema drift")
    source = frame.copy()
    source["ts"] = pd.to_datetime(source.ts, utc=True, errors="coerce")
    for column in ("open", "high", "low", "close"):
        source[column] = pd.to_numeric(source[column], errors="coerce")
    if source.ts.isna().any() or source.ts.duplicated().any():
        raise RuntimeError("HVERI invalid source key")
    prices = source[["open", "high", "low", "close"]]
    source["row_valid"] = (
        np.isfinite(prices).all(axis=1)
        & prices.gt(0).all(axis=1)
        & source.high.ge(prices[["open", "close"]].max(axis=1))
        & source.low.le(prices[["open", "close"]].min(axis=1))
        & source.high.ge(source.low)
    )
    source["minute_sq_return"] = np.square(np.log(source.close / source.open)).where(source.row_valid)
    return source.set_index("ts").sort_index()


def cross_side(signal: pd.Series) -> pd.Series:
    previous=signal.shift(1);side=np.zeros(len(signal),dtype=int)
    side[(signal>0)&(previous<=0)]=1;side[(signal<0)&(previous>=0)]=-1
    return pd.Series(side,index=signal.index)


def super_smoother(values: np.ndarray, valid: np.ndarray, period: int) -> np.ndarray:
    a1=math.exp(-1.414*math.pi/period);c2=2*a1*math.cos(1.414*math.pi/period);c3=-a1*a1;c1=1-c2-c3;output=np.full(len(values),np.nan);run=0
    for i,(value,is_valid) in enumerate(zip(values,valid)):
        if not is_valid or not math.isfinite(value):run=0;continue
        run+=1;output[i]=value if run<=3 else c1*(value+values[i-1])/2+c2*output[i-1]+c3*output[i-2]
    return output


def reversion_index(series: pd.Series, valid: pd.Series) -> pd.DataFrame:
    length=int(P["length"]);values=series.to_numpy(float);ok=valid.to_numpy(bool);delta_sum=np.full(len(series),np.nan);absolute=np.full(len(series),np.nan);ratio=np.full(len(series),np.nan);runs=np.zeros(len(series),dtype=int);run=0
    for i,(price,is_valid) in enumerate(zip(values,ok)):
        if not is_valid or not math.isfinite(price):run=0;continue
        run+=1;runs[i]=run
        if run<=length:continue
        deltas=np.diff(values[i-length:i+1]);delta_sum[i]=float(deltas.sum());absolute[i]=float(np.abs(deltas).sum())
        if absolute[i]>0:ratio[i]=delta_sum[i]/absolute[i]
    ratio_valid=np.isfinite(ratio);smooth=super_smoother(ratio,ratio_valid,P["smooth_period"]);trigger=super_smoother(ratio,ratio_valid,P["trigger_period"])
    return pd.DataFrame({"run_length":runs,"delta_sum":delta_sum,"absolute_delta_sum":absolute,"ratio":ratio,"smooth":smooth,"trigger":trigger,"spread":trigger-smooth},index=series.index)

def build_panel(raw: pd.DataFrame) -> pd.DataFrame:
    source=prepare(raw).reindex(pd.date_range(START,END,freq="1min",inclusive="left"));groups=source.groupby(source.index.floor("4h"),sort=True)
    bars=pd.DataFrame({"rows":groups.row_valid.sum(),"bar_open":groups.open.first(),"bar_high":groups.high.max(),"bar_low":groups.low.min(),"bar_close":groups.close.last(),"variation_component":groups.minute_sq_return.sum(min_count=240)})
    fields=["bar_open","bar_high","bar_low","bar_close"];bars["valid_bar"]=bars.rows.eq(240)&np.isfinite(bars[fields]).all(axis=1)&bars[fields].gt(0).all(axis=1);values=reversion_index(bars.bar_close,bars.valid_bar)
    for column in values:bars[column]=values[column]
    bars["entry_side"]=cross_side(bars.spread);bars["ratio_cross_side"]=cross_side(bars.ratio);bars["variation"]=np.sqrt(bars.variation_component.rolling(P["variation_hours"]//4,min_periods=P["variation_hours"]//4).sum());numeric=["delta_sum","absolute_delta_sum","ratio","smooth","trigger","spread","variation"];bars["source_valid"]=bars.valid_bar&bars.run_length.ge(P["warmup_observations"])&np.isfinite(bars[numeric]).all(axis=1)&bars.absolute_delta_sum.gt(0)&bars.variation.gt(0)
    panel=bars.reset_index(names="source_start");panel["feature_available_time"]=panel.source_start+pd.Timedelta("4h");panel["variation_rank"]=prior_rank(panel.variation.where(panel.source_valid));panel["eligible"]=panel.source_valid&panel.entry_side.ne(0)&panel.variation_rank.ge(P["variation_rank_min"]);return panel.loc[:,PANEL_COLUMNS]


def active(panel: pd.DataFrame, control: str = "primary"):
    if control not in ("primary",*CONTROLS):raise ValueError(control)
    used=panel.copy();valid=used.source_valid.eq(True);variation=used.variation_rank.ge(P["variation_rank_min"]);side=pd.to_numeric(used.entry_side,errors="coerce").fillna(0).astype(int);state=valid&side.ne(0)&variation
    if control=="no_variation_gate":state=valid&side.ne(0)
    elif control=="raw_ratio_zero_cross":side=pd.to_numeric(used.ratio_cross_side,errors="coerce").fillna(0).astype(int);state=valid&side.ne(0)&variation
    elif control=="one_bar_stale_cross":state=state.shift(1,fill_value=False);side=side.shift(1,fill_value=0)
    elif control=="direction_flip":side=-side
    return state&side.ne(0),side,used

def build_clock(panel: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    activity, side, used = active(panel, control)
    rows = []
    reserved_until: pd.Timestamp | None = None
    for index in panel.index[activity]:
        decision = pd.Timestamp(panel.at[index, "feature_available_time"])
        entry = decision + pd.Timedelta(minutes=P["entry_delay_minutes"])
        exit_time = entry + pd.Timedelta(hours=P["hold_hours"])
        if reserved_until is not None and entry < reserved_until:
            continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None:
            continue
        reserved_until = exit_time
        rows.append(
            {"candidate": prereg.POLICY_ID, "control": control, "split": split,
             "source_start": pd.Timestamp(used.at[index, "source_start"]),
             "feature_available_time": decision, "entry_time": entry, "exit_time": exit_time,
             "side": int(side.at[index]),
             **{column: bool(used.at[index, column]) if column == "eligible" else float(used.at[index, column])
                for column in CLOCK_COLUMNS[8:]}}
        )
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, float | int]:
    selected = clock[clock.split.eq(split)]
    if selected.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(selected.side.eq(1).sum())
    shorts = int(selected.side.eq(-1).sum())
    months = pd.to_datetime(selected.entry_time, utc=True).dt.strftime("%Y-%m").value_counts()
    return {"events": len(selected), "longs": longs, "shorts": shorts,
            "minority_side_share": min(longs, shorts) / len(selected),
            "max_month_share": int(months.max()) / len(selected)}


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
        raise RuntimeError("HVERI prereg drift")
    raw = load_source()
    panel = build_panel(raw)
    primary = build_clock(panel)
    controls = {name: build_clock(panel, name) for name in CONTROLS}
    splits = {name: primary[primary.split.eq(name)].copy() for name in SPLITS}
    immutable(PANEL, csv_gz(panel)); immutable(CLOCK, csv_gz(primary))
    for name, frame in controls.items(): immutable(CONTROL_DIR / f"{name}.csv.gz", csv_gz(frame))
    for name, frame in splits.items(): immutable(SPLIT_DIR / f"{name}.csv.gz", csv_gz(frame))
    source_core = {
        "protocol_version": "hveri_24_source_v1", "query": QUERY,
        "query_sha256": hashlib.sha256(QUERY.encode()).hexdigest(),
        "window": [START.isoformat(), END.isoformat()], "physical_rows": len(raw),
        "builder": {"path": str(BUILDER), "sha256": sha(BUILDER)},
        "panel": {"path": str(PANEL), "sha256": sha(PANEL), "rows": len(panel),
                  "valid_rows": int(panel.source_valid.sum())},
        "outcomes_opened": False, "gross9_rows_opened": False, "no_imputation": True,
    }
    manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}
    immutable(MANIFEST, json_bytes(manifest))
    support = {name: stats(primary, name) for name in SPLITS}
    checks = {key: passed for name, values in support.items() for key, passed in (
        (f"{name}_minimum_events", values["events"] >= GATES["minimum_events"][name]),
        (f"{name}_side_balance", values["minority_side_share"] >= GATES["minority_side_share_min"]),
        (f"{name}_month_concentration", values["max_month_share"] <= GATES["max_month_share"]),
    )}
    passed = all(checks.values())
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    core = {
        "protocol_version": "hveri_24_source_support_v1", "policy_id": prereg.POLICY_ID,
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA,
                            "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(MANIFEST), "sha256": sha(MANIFEST),
                            "manifest_hash": manifest["manifest_hash"]},
        "completed_preentry_sources_opened": True, "candidate_incidence_opened": True,
        "postentry_return_pnl_execution_price_opened": False, "funding_values_opened": False,
        "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "split_artifacts": {name: {"path": str(SPLIT_DIR / f"{name}.csv.gz"),
                                   "sha256": sha(SPLIT_DIR / f"{name}.csv.gz"), "rows": len(frame)}
                            for name, frame in splits.items()},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"),
                            "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(frame),
                            "promotion_authorized": False} for name, frame in controls.items()},
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    immutable(RESULT, json_bytes(result))
    return result


if __name__ == "__main__":
    print(json.dumps({"passed": run()["support_passed"], "result": str(RESULT)}))
