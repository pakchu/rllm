"""Build outcome-blind source support for frozen HVVSR-24."""
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

from training import preregister_high_volatility_variance_signature_reversal_relay as prereg

ENV_FILE = "/home/pakchu/rllm/.env"
START = pd.Timestamp("2023-04-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA = "af6a8bf5dfb895fb5e586632fcb4e19e1664cb68871c25e1a9e978045a14199d"
REG = prereg.build()
P = REG["policy"]
SPLITS = {name: tuple(map(pd.Timestamp, window)) for name, window in REG["stages"].items()}
GATES = REG["source_support_gates"]
CONTROLS = tuple(REG["diagnostic_controls"]["names"])
QUERY = """SELECT ts,open,high,low,close FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts"""
ROOT = Path("data/high_volatility_variance_signature_reversal_relay_sources_2023_2026")
PANEL = ROOT / "four_hour_states.csv.gz"
MANIFEST = ROOT / "manifest.json"
CLOCK = Path("data/high_volatility_variance_signature_reversal_relay_clocks_2023_2026.csv.gz")
SPLIT_DIR = Path("data/high_volatility_variance_signature_reversal_relay_split_clocks_2023_2026")
CONTROL_DIR = Path("data/high_volatility_variance_signature_reversal_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_variance_signature_reversal_relay_support_2026-08-11.json")
BUILDER = Path(__file__).relative_to(Path.cwd())
PANEL_COLUMNS = (
    "source_start", "feature_available_time", "source_valid", "bar_open", "bar_close",
    "four_hour_return", "last_five_minute_return", "one_minute_rv", "five_minute_rv",
    "signature_ratio", "variation", "signature_rank", "variation_rank",
    "signature_tail", "variation_tail", "joint_state", "onset", "entry_side", "eligible",
)
CLOCK_COLUMNS = (
    "candidate", "control", "split", "source_start", "feature_available_time",
    "entry_time", "exit_time", "side", "bar_close", "four_hour_return",
    "last_five_minute_return", "one_minute_rv", "five_minute_rv", "signature_ratio",
    "variation", "signature_rank", "variation_rank", "signature_tail", "variation_tail",
    "joint_state", "onset", "entry_side", "eligible",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def prior_rank(series: pd.Series, continuity_valid: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(float)
    valid = continuity_valid.to_numpy(bool)
    output = np.full(len(values), np.nan)
    history: list[float] = []
    for index, value in enumerate(values):
        if not valid[index]:
            history = []
            continue
        prior = np.asarray(history[-P["rank_history_decisions"] :], float)
        if math.isfinite(value) and len(prior) >= P["minimum_rank_history_decisions"]:
            output[index] = (np.sum(prior < value) + 0.5 * np.sum(prior == value)) / len(prior)
        if math.isfinite(value):
            history.append(float(value))
    return pd.Series(output, index=series.index)


def fresh_onset(state: pd.Series, source_valid: pd.Series) -> pd.Series:
    output = np.zeros(len(state), dtype=bool)
    previous_state = False
    has_previous = False
    for index, (active, valid) in enumerate(zip(state.to_numpy(bool), source_valid.to_numpy(bool))):
        if not valid:
            previous_state = False
            has_previous = False
            continue
        output[index] = has_previous and active and not previous_state
        previous_state = bool(active)
        has_previous = True
    return pd.Series(output, index=state.index)


def multiscale_states(source: pd.DataFrame) -> pd.DataFrame:
    """Exact trailing 24h close-to-close RV at one- and five-minute sampling."""
    minute_return = np.log(source.close / source.close.shift()).where(source.row_valid & source.row_valid.shift(fill_value=False))
    one_rv = minute_return.pow(2).rolling(1439, min_periods=1439).sum()
    groups = source.groupby(source.index.floor("5min"), sort=True)
    five = pd.DataFrame({"rows": groups.row_valid.sum(), "close": groups.close.last()})
    five["valid"] = five.rows.eq(5) & np.isfinite(five.close) & five.close.gt(0)
    five["return"] = np.log(five.close / five.close.shift()).where(five.valid & five.valid.shift(fill_value=False))
    five["rv"] = five["return"].pow(2).rolling(287, min_periods=287).sum()
    four_hour_starts = pd.date_range(START, END, freq="4h", inclusive="left")
    minute_ends = four_hour_starts + pd.Timedelta(minutes=239)
    five_ends = four_hour_starts + pd.Timedelta(minutes=235)
    return pd.DataFrame({
        "one_minute_rv": one_rv.reindex(minute_ends).to_numpy(),
        "five_minute_rv": five.rv.reindex(five_ends).to_numpy(),
        "last_five_minute_return": five["return"].reindex(five_ends).to_numpy(),
    }, index=four_hour_starts)


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
            return pd.read_sql_query(text(QUERY), connection, params={"start": START.to_pydatetime(), "end": END.to_pydatetime()})
    finally:
        database.dispose()


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.columns.tolist() != ["ts", "open", "high", "low", "close"]:
        raise RuntimeError("HVVSR source schema drift")
    source = frame.copy()
    source["ts"] = pd.to_datetime(source.ts, utc=True, errors="coerce")
    for column in ("open", "high", "low", "close"):
        source[column] = pd.to_numeric(source[column], errors="coerce")
    if source.ts.isna().any() or source.ts.duplicated().any():
        raise RuntimeError("HVVSR invalid source key")
    prices = source[["open", "high", "low", "close"]]
    source["row_valid"] = (
        np.isfinite(prices).all(axis=1) & prices.gt(0).all(axis=1)
        & source.high.ge(prices[["open", "close"]].max(axis=1))
        & source.low.le(prices[["open", "close"]].min(axis=1)) & source.high.ge(source.low)
    )
    return source.set_index("ts").sort_index()


def build_panel(raw: pd.DataFrame) -> pd.DataFrame:
    source = prepare(raw).reindex(pd.date_range(START, END, freq="1min", inclusive="left"))
    source["row_valid"] = source.row_valid.fillna(False).astype(bool)
    states = multiscale_states(source)
    groups = source.groupby(source.index.floor("4h"), sort=True)
    bars = pd.DataFrame({
        "rows": groups.row_valid.sum(), "bar_open": groups.open.first(), "bar_close": groups.close.last(),
    })
    bars["valid_bar"] = bars.rows.eq(240) & np.isfinite(bars[["bar_open", "bar_close"]]).all(axis=1) & bars[["bar_open", "bar_close"]].gt(0).all(axis=1)
    bars = bars.join(states)
    bars["four_hour_return"] = np.log(bars.bar_close / bars.bar_open)
    bars["signature_ratio"] = bars.one_minute_rv / bars.five_minute_rv
    bars["variation"] = np.sqrt(bars.five_minute_rv)
    fields = ["four_hour_return", "last_five_minute_return", "one_minute_rv", "five_minute_rv", "signature_ratio", "variation"]
    bars["source_valid"] = bars.valid_bar & np.isfinite(bars[fields]).all(axis=1) & bars.four_hour_return.ne(0) & bars.last_five_minute_return.ne(0) & bars.one_minute_rv.gt(0) & bars.five_minute_rv.gt(0) & bars.signature_ratio.gt(0)
    panel = bars.reset_index(names="source_start")
    panel["feature_available_time"] = panel.source_start + pd.Timedelta("4h")
    panel["signature_rank"] = prior_rank(panel.signature_ratio.where(panel.source_valid), panel.valid_bar)
    panel["variation_rank"] = prior_rank(panel.variation.where(panel.source_valid), panel.valid_bar)
    panel["signature_tail"] = panel.signature_rank.ge(P["signature_rank_min"])
    panel["variation_tail"] = panel.variation_rank.ge(P["variation_rank_min"])
    panel["joint_state"] = panel.source_valid & panel.signature_tail & panel.variation_tail
    panel["onset"] = fresh_onset(panel.joint_state, panel.source_valid)
    panel["entry_side"] = -np.sign(panel.four_hour_return).fillna(0).astype(int)
    panel["eligible"] = panel.onset & panel.entry_side.ne(0)
    return panel.loc[:, PANEL_COLUMNS]


def active(panel: pd.DataFrame, control: str = "primary"):
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    used = panel.copy()
    valid = used.source_valid.eq(True)
    side = pd.to_numeric(used.entry_side, errors="coerce").fillna(0).astype(int)
    state = used.eligible.eq(True) & side.ne(0)
    if control == "no_signature_gate":
        state = fresh_onset(valid & used.variation_tail, valid) & side.ne(0)
    elif control == "no_variation_gate":
        state = fresh_onset(valid & used.signature_tail, valid) & side.ne(0)
    elif control == "five_minute_return_continuation":
        side = np.sign(pd.to_numeric(used.last_five_minute_return, errors="coerce")).fillna(0).astype(int)
        state = used.eligible.eq(True) & side.ne(0)
    elif control == "one_bar_stale_onset":
        state = state.shift(1, fill_value=False)
        side = side.shift(1, fill_value=0)
    elif control == "direction_flip":
        side = -side
    elif control == "forced_long":
        side = pd.Series(1, index=side.index, dtype=int)
    return state & side.ne(0), side, used


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
        rows.append({
            "candidate": prereg.POLICY_ID, "control": control, "split": split,
            "source_start": pd.Timestamp(used.at[index, "source_start"]), "feature_available_time": decision,
            "entry_time": entry, "exit_time": exit_time, "side": int(side.at[index]),
            **{column: bool(used.at[index, column]) if column in ("signature_tail", "variation_tail", "joint_state", "onset", "eligible") else float(used.at[index, column]) for column in CLOCK_COLUMNS[8:]},
        })
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, float | int]:
    selected = clock[clock.split.eq(split)]
    if selected.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs, shorts = int(selected.side.eq(1).sum()), int(selected.side.eq(-1).sum())
    months = pd.to_datetime(selected.entry_time, utc=True).dt.strftime("%Y-%m").value_counts()
    return {"events": len(selected), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(selected), "max_month_share": int(months.max()) / len(selected)}


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
        raise RuntimeError("HVVSR prereg drift")
    raw = load_source()
    panel = build_panel(raw)
    primary = build_clock(panel)
    controls = {name: build_clock(panel, name) for name in CONTROLS}
    splits = {name: primary[primary.split.eq(name)].copy() for name in SPLITS}
    immutable(PANEL, csv_gz(panel)); immutable(CLOCK, csv_gz(primary))
    for name, frame in controls.items(): immutable(CONTROL_DIR / f"{name}.csv.gz", csv_gz(frame))
    for name, frame in splits.items(): immutable(SPLIT_DIR / f"{name}.csv.gz", csv_gz(frame))
    source_core = {
        "protocol_version": "hvvsr_24_source_v1", "query": QUERY, "query_sha256": hashlib.sha256(QUERY.encode()).hexdigest(),
        "window": [START.isoformat(), END.isoformat()], "physical_rows": len(raw),
        "builder": {"path": str(BUILDER), "sha256": sha(BUILDER)},
        "panel": {"path": str(PANEL), "sha256": sha(PANEL), "rows": len(panel), "valid_rows": int(panel.source_valid.sum())},
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
        "protocol_version": "hvvsr_24_source_support_v1", "policy_id": prereg.POLICY_ID,
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(MANIFEST), "sha256": sha(MANIFEST), "manifest_hash": manifest["manifest_hash"]},
        "completed_preentry_sources_opened": True, "candidate_incidence_opened": True,
        "postentry_return_pnl_execution_price_opened": False, "funding_values_opened": False, "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "split_artifacts": {name: {"path": str(SPLIT_DIR / f"{name}.csv.gz"), "sha256": sha(SPLIT_DIR / f"{name}.csv.gz"), "rows": len(frame)} for name, frame in splits.items()},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(frame), "promotion_authorized": False} for name, frame in controls.items()},
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    immutable(RESULT, json_bytes(result))
    return result


if __name__ == "__main__":
    print(json.dumps({"passed": run()["support_passed"], "result": str(RESULT)}))
