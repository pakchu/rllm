"""Outcome-blind source-support gate for frozen HVABDS-8."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_alt_breadth_diffusion_slope_relay as prereg


ENV_FILE = "/home/pakchu/rllm/.env"
PREREG_SHA = "4ad369076ceb6288d9da9a58d357d12638554df038f3eff8be506c145e63154b"
START = pd.Timestamp("2023-01-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
REGISTRATION = prereg.build()
POLICY = REGISTRATION["policy"]
SPLITS = {
    name: tuple(pd.Timestamp(value) for value in bounds)
    for name, bounds in REGISTRATION["stages"].items()
}
GATES = REGISTRATION["source_support_gates"]
CONTROLS = tuple(REGISTRATION["diagnostic_controls"]["names"])
SYMBOLS = ("BTCUSDT", *prereg.ALTS)
SOURCE_QUERY = """
SELECT date_trunc('hour', ts) AS hour_time, symbol,
       (array_agg(open ORDER BY ts))[1] AS hour_open,
       (array_agg(close ORDER BY ts DESC))[1] AS hour_close,
       sum(power(ln(close/open),2)) AS squared_variation,
       count(*) AS source_rows, count(DISTINCT ts) AS distinct_rows,
       min(ts) AS first_ts, max(ts) AS last_ts,
       bool_and(open>0 AND high>0 AND low>0 AND close>0
                AND high>=greatest(open,close,low)
                AND low<=least(open,close,high)) AS coherent
FROM bars_binance
WHERE symbol=ANY(:symbols) AND interval='1m' AND ts>=:start AND ts<:end
GROUP BY 1,2 ORDER BY 1,2
"""
SOURCE_DIR = Path("data/high_volatility_alt_breadth_diffusion_slope_relay_sources_2023_2026")
PANEL = SOURCE_DIR / "eight_hour_states.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_alt_breadth_diffusion_slope_relay_clocks_2023_2026.csv.gz")
SPLIT_DIR = Path("data/high_volatility_alt_breadth_diffusion_slope_relay_split_clocks_2023_2026")
CONTROL_DIR = Path("data/high_volatility_alt_breadth_diffusion_slope_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_alt_breadth_diffusion_slope_relay_support_2026-08-13.json")
BUILDER = Path(__file__).relative_to(Path.cwd())
PANEL_COLUMNS = (
    "decision_time", "feature_available_time", "source_valid", "aggregate_alt_return",
    "side", "participation_0", "participation_1", "participation_2", "participation_3",
    "participation_4", "participation_5", "participation_6", "participation_7",
    "diffusion_slope", "slope_rank", "btc_realized_variation", "variation_rank",
    "endpoint_breadth", "eligible", "onset",
)
CLOCK_COLUMNS = (
    "candidate", "control", "split", "decision_time", "feature_available_time",
    "entry_time", "exit_time", "side", *PANEL_COLUMNS[3:5], *PANEL_COLUMNS[5:18],
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return prereg.canonical_hash(value)


def strict_prior_midrank(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    output = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        prior = np.asarray(history[-POLICY["history_decisions"] :], dtype=float)
        if math.isfinite(current) and len(prior) >= POLICY["minimum_history_decisions"]:
            output.at[index] = (
                np.sum(prior < current) + 0.5 * np.sum(prior == current)
            ) / len(prior)
        if math.isfinite(current):
            history.append(float(current))
    return output


def participation_slope(shares: np.ndarray) -> float:
    values = np.asarray(shares, dtype=float)
    if values.shape != (8,) or not np.isfinite(values).all():
        return math.nan
    coordinates = np.arange(8, dtype=float)
    centered = coordinates - coordinates.mean()
    return float(np.dot(centered, values - values.mean()) / np.dot(centered, centered))


def previous_valid_onset(eligible: pd.Series, source_valid: pd.Series) -> pd.Series:
    output = pd.Series(False, index=eligible.index)
    previous = None
    for index in eligible.index:
        if not bool(source_valid.at[index]):
            continue
        if bool(eligible.at[index]) and previous is not None:
            output.at[index] = not bool(eligible.at[previous])
        previous = index
    return output


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
                text(SOURCE_QUERY), connection,
                params={"symbols": list(SYMBOLS), "start": START, "end": END},
            )
    finally:
        database.dispose()


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    required = {
        "hour_time", "symbol", "hour_open", "hour_close", "squared_variation",
        "source_rows", "distinct_rows", "first_ts", "last_ts", "coherent",
    }
    if set(raw.columns) != required:
        raise RuntimeError("HVABDS source schema drift")
    frame = raw.copy()
    for column in ("hour_time", "first_ts", "last_ts"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    for column in (
        "hour_open", "hour_close", "squared_variation", "source_rows", "distinct_rows"
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["hour_time", "symbol"]].isna().any().any() or frame.duplicated(["hour_time", "symbol"]).any():
        raise RuntimeError("HVABDS invalid source key")
    frame["hour_return"] = np.log(frame["hour_close"] / frame["hour_open"])
    frame["row_valid"] = (
        frame["symbol"].isin(SYMBOLS)
        & np.isfinite(frame[["hour_open", "hour_close", "squared_variation"]]).all(axis=1)
        & frame["hour_open"].gt(0)
        & frame["hour_close"].gt(0)
        & frame["squared_variation"].ge(0)
        & frame["source_rows"].eq(60)
        & frame["distinct_rows"].eq(60)
        & frame["first_ts"].eq(frame["hour_time"])
        & frame["last_ts"].eq(frame["hour_time"] + pd.Timedelta(minutes=59))
        & frame["coherent"].eq(True)
    )
    return frame.set_index(["hour_time", "symbol"]).sort_index()


def build_panel(raw: pd.DataFrame) -> pd.DataFrame:
    source = prepare(raw)
    rows: list[dict[str, Any]] = []
    first_decision = START + pd.Timedelta(hours=27)
    for decision in pd.date_range(first_decision, END, freq="8h", inclusive="left"):
        block_hours = pd.date_range(decision - pd.Timedelta(hours=8), decision, freq="1h", inclusive="left")
        variation_hours = pd.date_range(decision - pd.Timedelta(hours=24), decision, freq="1h", inclusive="left")
        block_index = pd.MultiIndex.from_product([block_hours, SYMBOLS], names=["hour_time", "symbol"])
        block = source.reindex(block_index)
        btc_index = pd.MultiIndex.from_product([variation_hours, ["BTCUSDT"]], names=["hour_time", "symbol"])
        btc = source.reindex(btc_index)
        valid = bool(
            len(block) == 8 * len(SYMBOLS)
            and block["row_valid"].eq(True).all()
            and len(btc) == 24
            and btc["row_valid"].eq(True).all()
        )
        aggregate = slope = variation = math.nan
        side = endpoint_breadth = 0
        shares = np.full(8, np.nan)
        if valid:
            alt = block.loc[(slice(None), list(prereg.ALTS)), "hour_return"].unstack("symbol")
            aggregate = float(alt.to_numpy(float).sum())
            side = int(np.sign(aggregate))
            if side:
                shares = (np.sign(alt.to_numpy(float)) == side).mean(axis=1)
                slope = participation_slope(shares)
                endpoint_breadth = int((np.sign(alt.sum(axis=0).to_numpy(float)) == side).sum())
            variation = float(math.sqrt(btc["squared_variation"].to_numpy(float).sum()))
            valid = bool(
                side != 0 and math.isfinite(slope) and math.isfinite(variation) and variation > 0
            )
        row = {
            "decision_time": decision, "feature_available_time": decision,
            "source_valid": valid, "aggregate_alt_return": aggregate, "side": side,
            **{f"participation_{i}": float(shares[i]) for i in range(8)},
            "diffusion_slope": slope, "btc_realized_variation": variation,
            "endpoint_breadth": endpoint_breadth,
        }
        rows.append(row)
    panel = pd.DataFrame(rows)
    valid = panel["source_valid"].eq(True)
    panel["slope_rank"] = strict_prior_midrank(panel["diffusion_slope"].where(valid))
    panel["variation_rank"] = strict_prior_midrank(panel["btc_realized_variation"].where(valid))
    panel["eligible"] = (
        valid & panel["diffusion_slope"].gt(0)
        & panel["slope_rank"].ge(POLICY["slope_rank_min"])
        & panel["variation_rank"].ge(POLICY["variation_rank_min"])
    )
    panel["onset"] = previous_valid_onset(panel["eligible"], valid)
    return panel.loc[:, PANEL_COLUMNS]


def active(panel: pd.DataFrame, control: str = "primary"):
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    used = panel.copy()
    if control == "one_block_stale_geometry":
        columns = [
            "source_valid", "side", "diffusion_slope", "slope_rank", "variation_rank",
            "endpoint_breadth", "feature_available_time",
        ]
        used[columns] = panel[columns].shift(1)
    valid = used["source_valid"].eq(True)
    slope_tail = used["diffusion_slope"].gt(0) & used["slope_rank"].ge(POLICY["slope_rank_min"])
    variation = used["variation_rank"].ge(POLICY["variation_rank_min"])
    state = valid & slope_tail & variation
    if control == "no_slope_tail_gate":
        state = valid & used["diffusion_slope"].gt(0) & variation
    elif control == "no_variation_gate":
        state = valid & slope_tail
    elif control == "endpoint_static_breadth":
        state = valid & used["endpoint_breadth"].ge(4) & variation
    onset = previous_valid_onset(state, valid)
    side = pd.to_numeric(used["side"], errors="coerce").fillna(0).astype(int)
    if control == "direction_flip":
        side = -side
    elif control == "forced_long":
        side = side.where(side.eq(0), 1)
    return onset & side.ne(0), side, used


def build_clock(panel: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    activated, side, used = active(panel, control)
    rows: list[dict[str, Any]] = []
    reserved_until = None
    for index in panel.index[activated]:
        decision = pd.Timestamp(panel.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=POLICY["entry_delay_minutes"])
        exit_time = entry + pd.Timedelta(hours=POLICY["hold_hours"])
        if reserved_until is not None and entry < reserved_until:
            continue
        split = next(
            (name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end),
            None,
        )
        if split is None:
            continue
        reserved_until = exit_time
        rows.append({
            "candidate": prereg.POLICY_ID, "control": control, "split": split,
            "decision_time": decision,
            "feature_available_time": pd.Timestamp(used.at[index, "feature_available_time"]),
            "entry_time": entry, "exit_time": exit_time, "side": int(side.at[index]),
            **{column: float(used.at[index, column]) for column in CLOCK_COLUMNS[8:]},
        })
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def support_stats(clock: pd.DataFrame, split: str) -> dict[str, float | int]:
    selected = clock.loc[clock["split"].eq(split)]
    if selected.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(selected["side"].eq(1).sum())
    shorts = int(selected["side"].eq(-1).sum())
    months = pd.to_datetime(selected["entry_time"], utc=True).dt.strftime("%Y-%m").value_counts()
    return {
        "events": len(selected), "longs": longs, "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(selected),
        "max_month_share": int(months.max()) / len(selected),
    }


def gzip_csv(frame: pd.DataFrame) -> bytes:
    raw = frame.to_csv(index=False, float_format="%.12g", lineterminator="\n").encode()
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=6, mtime=0) as archive:
        archive.write(raw)
    return buffer.getvalue()


def immutable_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != content:
        raise RuntimeError(f"refusing to overwrite immutable artifact {path}")
    path.write_bytes(content)


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode()


def run() -> dict[str, Any]:
    if sha256(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVABDS preregistration drift")
    raw = load_source()
    panel = build_panel(raw)
    primary = build_clock(panel)
    controls = {name: build_clock(panel, name) for name in CONTROLS}
    splits = {name: primary.loc[primary["split"].eq(name)].copy() for name in SPLITS}
    immutable_write(PANEL, gzip_csv(panel))
    immutable_write(CLOCK, gzip_csv(primary))
    for name, frame in controls.items():
        immutable_write(CONTROL_DIR / f"{name}.csv.gz", gzip_csv(frame))
    for name, frame in splits.items():
        immutable_write(SPLIT_DIR / f"{name}.csv.gz", gzip_csv(frame))
    source_core = {
        "protocol_version": "hvabds_8_sources_v1",
        "query": SOURCE_QUERY,
        "query_sha256": hashlib.sha256(SOURCE_QUERY.encode()).hexdigest(),
        "tables": ["bars_binance"], "symbols": list(SYMBOLS),
        "window": [START.isoformat(), END.isoformat()], "physical_rows": len(raw),
        "builder": {"path": str(BUILDER), "sha256": sha256(BUILDER)},
        "panel": {"path": str(PANEL), "sha256": sha256(PANEL), "rows": len(panel), "valid_rows": int(panel["source_valid"].sum())},
        "outcomes_opened": False, "gross9_rows_opened": False,
        "no_imputation": True,
    }
    source_manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}
    immutable_write(SOURCE_MANIFEST, json_bytes(source_manifest))
    support = {name: support_stats(primary, name) for name in SPLITS}
    checks = {
        key: passed
        for name, stats in support.items()
        for key, passed in (
            (f"{name}_minimum_events", stats["events"] >= GATES["minimum_events"][name]),
            (f"{name}_side_balance", stats["minority_side_share"] >= GATES["minority_side_share_min"]),
            (f"{name}_month_concentration", stats["max_month_share"] <= GATES["max_month_share"]),
        )
    }
    passed = all(checks.values())
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    core = {
        "protocol_version": "hvabds_8_source_support_v1", "policy_id": prereg.POLICY_ID,
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(SOURCE_MANIFEST), "sha256": sha256(SOURCE_MANIFEST), "manifest_hash": source_manifest["manifest_hash"]},
        "completed_preentry_sources_opened": True, "candidate_incidence_opened": True,
        "postentry_return_pnl_execution_price_opened": False, "funding_values_opened": False,
        "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha256(CLOCK), "rows": len(primary)},
        "split_artifacts": {name: {"path": str(SPLIT_DIR / f"{name}.csv.gz"), "sha256": sha256(SPLIT_DIR / f"{name}.csv.gz"), "rows": len(frame)} for name, frame in splits.items()},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha256(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(frame), "promotion_authorized": False} for name, frame in controls.items()},
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    immutable_write(RESULT, json_bytes(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    result = run()
    print(json.dumps({"passed": result["support_passed"], "support": result["support"], "result": str(RESULT)}))


if __name__ == "__main__":
    main()
