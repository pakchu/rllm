"""Build outcome-blind source support for frozen HVRDV-8."""
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

from training import preregister_high_volatility_relative_daily_volume_continuation_relay as prereg

ENV_FILE = "/home/pakchu/rllm/.env"
START = pd.Timestamp("2023-01-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA = "7bc629565ae92c6ff5b8e6e00797f816312facaf2aa962bc20d81db718606f81"
REG = prereg.build()
P = REG["policy"]
SPLITS = {name: tuple(map(pd.Timestamp, window)) for name, window in REG["stages"].items()}
GATES = REG["source_support_gates"]
CONTROLS = tuple(REG["diagnostic_controls"]["names"])
QUERY = """SELECT ts,open,high,low,close,volume,quote_asset_volume FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts"""
ROOT = Path("data/high_volatility_relative_daily_volume_continuation_relay_sources_2023_2026")
PANEL = ROOT / "four_hour_states.csv.gz"
MANIFEST = ROOT / "manifest.json"
CLOCK = Path("data/high_volatility_relative_daily_volume_continuation_relay_clocks_2023_2026.csv.gz")
SPLIT_DIR = Path("data/high_volatility_relative_daily_volume_continuation_relay_split_clocks_2023_2026")
CONTROL_DIR = Path("data/high_volatility_relative_daily_volume_continuation_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_relative_daily_volume_continuation_relay_support_2026-08-11.json")
BUILDER = Path(__file__).relative_to(Path.cwd())
PANEL_COLUMNS = (
    "source_start", "source_day", "elapsed_hours", "feature_available_time",
    "source_valid", "bar_open", "bar_close", "bar_volume", "bar_quote_volume",
    "cumulative_volume", "historical_average_volume", "rdv",
    "cumulative_quote_volume", "historical_average_quote_volume", "quote_rdv",
    "day_return", "variation", "variation_rank", "onset", "quote_onset", "eligible",
)
CLOCK_COLUMNS = (
    "candidate", "control", "split", "source_start", "feature_available_time",
    "entry_time", "exit_time", "side", "elapsed_hours", "bar_volume",
    "cumulative_volume", "historical_average_volume", "rdv", "quote_rdv",
    "day_return", "variation", "variation_rank", "onset", "eligible",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def prior_rank(series: pd.Series, continuity_valid: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(float)
    output = np.full(len(values), np.nan)
    history: list[float] = []
    valid = continuity_valid.to_numpy(bool)
    for index, value in enumerate(values):
        if not valid[index]:
            history = []
            continue
        prior = np.asarray(history[-P["variation_history_slots"] :], float)
        if math.isfinite(value) and len(prior) >= P["minimum_variation_history_slots"]:
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
    if frame.columns.tolist() != ["ts", "open", "high", "low", "close", "volume", "quote_asset_volume"]:
        raise RuntimeError("HVRDV source schema drift")
    source = frame.copy()
    source["ts"] = pd.to_datetime(source.ts, utc=True, errors="coerce")
    for column in ("open", "high", "low", "close", "volume", "quote_asset_volume"):
        source[column] = pd.to_numeric(source[column], errors="coerce")
    if source.ts.isna().any() or source.ts.duplicated().any():
        raise RuntimeError("HVRDV invalid source key")
    prices = source[["open", "high", "low", "close"]]
    source["row_valid"] = (
        np.isfinite(prices).all(axis=1)
        & prices.gt(0).all(axis=1)
        & source.high.ge(prices[["open", "close"]].max(axis=1))
        & source.low.le(prices[["open", "close"]].min(axis=1))
        & source.high.ge(source.low)
        & source.volume.ge(0)
        & source.quote_asset_volume.ge(0)
    )
    source["minute_sq_return"] = np.square(np.log(source.close / source.open)).where(source.row_valid)
    return source.set_index("ts").sort_index()


def relative_daily_volume(bars: pd.DataFrame, volume_column: str) -> pd.DataFrame:
    """Compute causal same-slot cumulative volume with reset after an invalid slot."""
    period = P["relative_daily_volume_days"]
    cumulative = np.full(len(bars), np.nan)
    average = np.full(len(bars), np.nan)
    ratio = np.full(len(bars), np.nan)
    onset = np.zeros(len(bars), dtype=bool)
    history: list[list[float]] = []
    day_values: list[float] = []
    running = 0.0
    previous_ratio = 0.0
    day_valid = True
    previous_day = None
    for index, (timestamp, row) in enumerate(bars.iterrows()):
        day = timestamp.floor("1d")
        if previous_day is not None and day != previous_day:
            if day_valid and len(day_values) == 6:
                history.append(day_values)
                history = history[-period:]
            else:
                history = []
            day_values = []
            running = 0.0
            previous_ratio = 0.0
            day_valid = True
        previous_day = day
        amount = float(row[volume_column])
        if not bool(row.valid_bar) or not math.isfinite(amount):
            day_valid = False
            history = []
            day_values = []
            running = 0.0
            previous_ratio = np.nan
            continue
        if not day_valid:
            continue
        running += amount
        day_values.append(running)
        cumulative[index] = running
        slot = len(day_values) - 1
        if len(history) == period:
            denominator = float(np.mean([values[slot] for values in history]))
            if denominator > 0:
                average[index] = denominator
                ratio[index] = running / denominator
                onset[index] = ratio[index] > 1.0 and previous_ratio <= 1.0
                previous_ratio = ratio[index]
    return pd.DataFrame(
        {"cumulative": cumulative, "average": average, "ratio": ratio, "onset": onset},
        index=bars.index,
    )


def build_panel(raw: pd.DataFrame) -> pd.DataFrame:
    source=prepare(raw).reindex(pd.date_range(START,END,freq="1min",inclusive="left"));groups=source.groupby(source.index.floor("4h"),sort=True)
    bars=pd.DataFrame({"rows":groups.row_valid.sum(),"bar_open":groups.open.first(),"bar_close":groups.close.last(),"bar_volume":groups.volume.sum(min_count=240),"bar_quote_volume":groups.quote_asset_volume.sum(min_count=240),"variation_component":groups.minute_sq_return.sum(min_count=240)})
    fields=["bar_open","bar_close","bar_volume","bar_quote_volume"];bars["valid_bar"]=bars.rows.eq(240)&np.isfinite(bars[fields]).all(axis=1)&bars[["bar_open","bar_close"]].gt(0).all(axis=1)&bars[["bar_volume","bar_quote_volume"]].gt(0).all(axis=1)
    base=relative_daily_volume(bars,"bar_volume");quote=relative_daily_volume(bars,"bar_quote_volume")
    for target, source_column in (
        ("cumulative_volume", "cumulative"),
        ("historical_average_volume", "average"),
        ("rdv", "ratio"),
        ("onset", "onset"),
    ):
        bars[target] = base[source_column]
    for target, source_column in (
        ("cumulative_quote_volume", "cumulative"),
        ("historical_average_quote_volume", "average"),
        ("quote_rdv", "ratio"),
        ("quote_onset", "onset"),
    ):
        bars[target] = quote[source_column]
    source_day=bars.index.floor("1d");day_open=bars.bar_open.groupby(source_day).transform("first");bars["day_return"]=np.log(bars.bar_close/day_open)
    bars["variation"]=np.sqrt(bars.variation_component.rolling(6,min_periods=6).sum());bars["source_valid"]=bars.valid_bar&np.isfinite(bars[["rdv","quote_rdv","day_return","variation"]]).all(axis=1)&bars.day_return.ne(0)&bars.variation.gt(0)
    panel=bars.reset_index(names="source_start");panel["source_day"]=panel.source_start.dt.floor("1d");panel["elapsed_hours"]=(panel.source_start.dt.hour+4).astype(int);panel["feature_available_time"]=panel.source_start+pd.Timedelta("4h");panel["variation_rank"]=prior_rank(panel.variation.where(panel.source_valid),panel.valid_bar);panel["eligible"]=panel.source_valid&panel.onset&panel.variation_rank.ge(P["variation_rank_min"]);return panel.loc[:,PANEL_COLUMNS]


def active(panel: pd.DataFrame, control: str = "primary"):
    if control not in ("primary", *CONTROLS): raise ValueError(control)
    used=panel.copy();valid=used.source_valid.eq(True);variation=used.variation_rank.ge(P["variation_rank_min"]);side=np.sign(pd.to_numeric(used.day_return,errors="coerce")).fillna(0).astype(int);state=valid&side.ne(0)&used.onset&variation
    if control=="no_variation_gate":state=valid&side.ne(0)&used.onset
    elif control=="no_rdv_onset_gate":state=valid&side.ne(0)&variation
    elif control=="quote_volume_rdv":state=valid&side.ne(0)&used.quote_onset&variation
    elif control=="one_slot_stale_onset":state=state.shift(1,fill_value=False);side=side.shift(1,fill_value=0)
    elif control=="direction_flip":side=-side
    elif control=="forced_long":side=pd.Series(1,index=side.index,dtype=int)
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
             **{column: bool(used.at[index, column]) if column in ("onset","eligible") else float(used.at[index, column])
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
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVRDV prereg drift")
    raw = load_source()
    panel = build_panel(raw)
    primary = build_clock(panel)
    controls = {name: build_clock(panel, name) for name in CONTROLS}
    splits = {name: primary[primary.split.eq(name)].copy() for name in SPLITS}
    immutable(PANEL, csv_gz(panel)); immutable(CLOCK, csv_gz(primary))
    for name, frame in controls.items(): immutable(CONTROL_DIR / f"{name}.csv.gz", csv_gz(frame))
    for name, frame in splits.items(): immutable(SPLIT_DIR / f"{name}.csv.gz", csv_gz(frame))
    source_core = {
        "protocol_version": "hvrdv_8_source_v1", "query": QUERY,
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
        "protocol_version": "hvrdv_8_source_support_v1", "policy_id": prereg.POLICY_ID,
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
