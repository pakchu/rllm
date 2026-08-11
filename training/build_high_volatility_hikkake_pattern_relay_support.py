"""Build outcome-blind source support for frozen HVHIKKAKE-C3-8."""
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

from training import preregister_high_volatility_hikkake_pattern_relay as prereg


ENV_FILE = "/home/pakchu/rllm/.env"
START = pd.Timestamp("2023-04-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA = "5e8a930d33856288cb670770da33cd85a1a9009377d4ad268cf0a13e49258704"
REG = prereg.build()
P = REG["policy"]
SPLITS = {name: tuple(map(pd.Timestamp, window)) for name, window in REG["stages"].items()}
GATES = REG["source_support_gates"]
CONTROLS = tuple(REG["diagnostic_controls"]["names"])
QUERY = """SELECT ts,open,high,low,close FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts"""
ROOT = Path("data/high_volatility_hikkake_pattern_relay_sources_2023_2026")
PANEL = ROOT / "hourly_states.csv.gz"
MANIFEST = ROOT / "manifest.json"
CLOCK = Path("data/high_volatility_hikkake_pattern_relay_clocks_2023_2026.csv.gz")
SPLIT_DIR = Path("data/high_volatility_hikkake_pattern_relay_split_clocks_2023_2026")
CONTROL_DIR = Path("data/high_volatility_hikkake_pattern_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_hikkake_pattern_relay_support_2026-08-11.json")
BUILDER = Path(__file__).relative_to(Path.cwd())
PANEL_COLUMNS = (
    "source_start", "feature_available_time", "source_valid", "bar_open", "bar_high",
    "bar_low", "bar_close", "hikkake_ready", "hikkake_output", "entry_side",
    "initial_setup_side", "all_nonzero_side", "variation", "variation_rank", "eligible",
)
CLOCK_COLUMNS = (
    "candidate", "control", "split", "source_start", "feature_available_time",
    "entry_time", "exit_time", "side", "hikkake_ready", "hikkake_output",
    "entry_side", "initial_setup_side", "all_nonzero_side", "variation",
    "variation_rank", "eligible",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def prior_rank(series: pd.Series, continuity_valid: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(float)
    valid = continuity_valid.to_numpy(bool)
    output = np.full(len(values), np.nan)
    history: list[float] = []
    for index, value in enumerate(values):
        if not valid[index]:
            history = []
            continue
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
                text(QUERY),
                connection,
                params={"start": START.to_pydatetime(), "end": END.to_pydatetime()},
            )
    finally:
        database.dispose()


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.columns.tolist() != ["ts", "open", "high", "low", "close"]:
        raise RuntimeError("HVHIKKAKE source schema drift")
    source = frame.copy()
    source["ts"] = pd.to_datetime(source.ts, utc=True, errors="coerce")
    for column in ("open", "high", "low", "close"):
        source[column] = pd.to_numeric(source[column], errors="coerce")
    if source.ts.isna().any() or source.ts.duplicated().any():
        raise RuntimeError("HVHIKKAKE invalid source key")
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


def hikkake_outputs(bars: pd.DataFrame, valid: pd.Series) -> pd.DataFrame:
    """Replicate LEAN Hikkake readiness, setup precedence, and confirmation outputs."""
    output = np.zeros(len(bars), dtype=int)
    ready = np.zeros(len(bars), dtype=bool)
    samples = 0
    history: list[tuple[float, float]] = []
    pattern_result = 0
    pattern_sample = 0
    inside_high = math.nan
    inside_low = math.nan
    for index, is_valid in enumerate(valid.to_numpy(bool)):
        if not is_valid:
            samples = 0
            history = []
            pattern_result = 0
            pattern_sample = 0
            inside_high = inside_low = math.nan
            continue
        current = bars.iloc[index]
        samples += 1
        history.append((float(current.bar_high), float(current.bar_low)))
        history = history[-3:]
        ready[index] = samples >= P["indicator_period_hours"]
        value = 0
        new_pattern = False
        if len(history) >= 3:
            first_high, first_low = history[-3]
            second_high, second_low = history[-2]
            current_high, current_low = history[-1]
            inside = second_high < first_high and second_low > first_low
            bullish = current_high < second_high and current_low < second_low
            bearish = current_high > second_high and current_low > second_low
            if inside and (bullish or bearish):
                pattern_result = 1 if bullish else -1
                pattern_sample = samples
                inside_high = second_high
                inside_low = second_low
                value = pattern_result
                new_pattern = True
        if not new_pattern and pattern_result and samples <= pattern_sample + P["confirmation_bars"]:
            close = float(current.bar_close)
            confirmed = (pattern_result > 0 and close > inside_high) or (
                pattern_result < 0 and close < inside_low
            )
            if confirmed:
                value = pattern_result + (1 if pattern_result > 0 else -1)
                pattern_result = 0
                pattern_sample = 0
                inside_high = inside_low = math.nan
        output[index] = value if ready[index] else 0
    initial = np.where(np.abs(output) == 1, np.sign(output), 0).astype(int)
    entry = np.where(np.abs(output) == P["confirmation_absolute_output"], np.sign(output), 0).astype(int)
    all_nonzero = np.sign(output).astype(int)
    return pd.DataFrame(
        {
            "hikkake_ready": ready,
            "hikkake_output": output,
            "entry_side": entry,
            "initial_setup_side": initial,
            "all_nonzero_side": all_nonzero,
        },
        index=bars.index,
    )


def build_panel(raw: pd.DataFrame) -> pd.DataFrame:
    minute_index = pd.date_range(START, END, freq="1min", inclusive="left")
    source = prepare(raw).reindex(minute_index)
    groups = source.groupby(source.index.floor("1h"), sort=True)
    bars = pd.DataFrame(
        {
            "rows": groups.row_valid.sum(),
            "bar_open": groups.open.first(),
            "bar_high": groups.high.max(),
            "bar_low": groups.low.min(),
            "bar_close": groups.close.last(),
            "variation_component": groups.minute_sq_return.sum(min_count=60),
        }
    )
    fields = ["bar_open", "bar_high", "bar_low", "bar_close"]
    bars["valid_bar"] = (
        bars.rows.eq(60)
        & np.isfinite(bars[fields]).all(axis=1)
        & bars[fields].gt(0).all(axis=1)
        & bars.bar_high.ge(bars[["bar_open", "bar_close"]].max(axis=1))
        & bars.bar_low.le(bars[["bar_open", "bar_close"]].min(axis=1))
        & bars.bar_high.ge(bars.bar_low)
    )
    bars = bars.join(hikkake_outputs(bars, bars.valid_bar))
    bars["variation"] = np.sqrt(
        bars.variation_component.rolling(P["variation_hours"], min_periods=P["variation_hours"]).sum()
    )
    bars["source_valid"] = (
        bars.valid_bar
        & bars.hikkake_ready
        & np.isfinite(bars.variation)
        & bars.variation.gt(0)
    )
    panel = bars.reset_index(names="source_start")
    panel["feature_available_time"] = panel.source_start + pd.Timedelta("1h")
    panel["variation_rank"] = prior_rank(
        panel.variation.where(panel.source_valid), panel.valid_bar
    )
    panel["eligible"] = (
        panel.source_valid
        & panel.entry_side.ne(0)
        & panel.variation_rank.ge(P["variation_rank_min"])
    )
    return panel.loc[:, PANEL_COLUMNS]


def active(panel: pd.DataFrame, control: str = "primary"):
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    used = panel.copy()
    valid = used.source_valid.eq(True)
    variation = used.variation_rank.ge(P["variation_rank_min"])
    side = pd.to_numeric(used.entry_side, errors="coerce").fillna(0).astype(int)
    state = valid & side.ne(0) & variation
    if control == "no_variation_gate":
        state = valid & side.ne(0)
    elif control == "initial_setup_only":
        side = pd.to_numeric(used.initial_setup_side, errors="coerce").fillna(0).astype(int)
        state = valid & side.ne(0) & variation
    elif control == "all_nonzero_hikkake_outputs":
        side = pd.to_numeric(used.all_nonzero_side, errors="coerce").fillna(0).astype(int)
        state = valid & side.ne(0) & variation
    elif control == "one_hour_stale_confirmation":
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
        split = next(
            (name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end),
            None,
        )
        if split is None:
            continue
        reserved_until = exit_time
        rows.append(
            {
                "candidate": prereg.POLICY_ID,
                "control": control,
                "split": split,
                "source_start": pd.Timestamp(used.at[index, "source_start"]),
                "feature_available_time": decision,
                "entry_time": entry,
                "exit_time": exit_time,
                "side": int(side.at[index]),
                **{
                    column: bool(used.at[index, column])
                    if column in ("hikkake_ready", "eligible")
                    else float(used.at[index, column])
                    for column in CLOCK_COLUMNS[8:]
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
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVHIKKAKE prereg drift")
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
        "protocol_version": "hvhikkake_c3_8_source_v1",
        "query": QUERY,
        "query_sha256": hashlib.sha256(QUERY.encode()).hexdigest(),
        "window": [START.isoformat(), END.isoformat()],
        "physical_rows": len(raw),
        "builder": {"path": str(BUILDER), "sha256": sha(BUILDER)},
        "panel": {
            "path": str(PANEL),
            "sha256": sha(PANEL),
            "rows": len(panel),
            "valid_rows": int(panel.source_valid.sum()),
        },
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
        "protocol_version": "hvhikkake_c3_8_source_support_v1",
        "policy_id": prereg.POLICY_ID,
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT),
            "sha256": PREREG_SHA,
            "manifest_hash": registration["manifest_hash"],
        },
        "source_manifest": {
            "path": str(MANIFEST),
            "sha256": sha(MANIFEST),
            "manifest_hash": manifest["manifest_hash"],
        },
        "completed_preentry_sources_opened": True,
        "candidate_incidence_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "funding_values_opened": False,
        "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "split_artifacts": {
            name: {
                "path": str(SPLIT_DIR / f"{name}.csv.gz"),
                "sha256": sha(SPLIT_DIR / f"{name}.csv.gz"),
                "rows": len(frame),
            }
            for name, frame in splits.items()
        },
        "controls": {
            name: {
                "path": str(CONTROL_DIR / f"{name}.csv.gz"),
                "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"),
                "rows": len(frame),
                "promotion_authorized": False,
            }
            for name, frame in controls.items()
        },
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
