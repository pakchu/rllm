"""Build source-only FETLS-6 clocks without opening execution outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_fast_empty_template_loss_shock as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
from training.download_fetls_block_summaries import OUTPUT as BLOCKS


ENV_FILE = "/home/pakchu/rllm/.env"
PREREG_SHA = "2d1cdc7898791a821e6de5960b2c7b6333d6b5f526ed6e9e1547349aa7bf577c"
START = pd.Timestamp("2023-07-01T00:00Z")
END = pd.Timestamp("2026-08-01T00:00Z")
SPLITS = {
    "train": (START, pd.Timestamp("2024-01-01T00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00Z"), pd.Timestamp("2025-01-01T00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00Z"), pd.Timestamp("2026-01-01T00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00Z"), END),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
BAR_QUERY = """SELECT date_bin('5 minutes',ts,TIMESTAMPTZ '1970-01-01 00:00:00+00') AS bar_time,(array_agg(open ORDER BY ts))[1] AS bar_open,(array_agg(close ORDER BY ts DESC))[1] AS bar_close,count(*) AS source_rows,count(DISTINCT ts) AS distinct_rows,min(ts) AS first_ts,max(ts) AS last_ts,bool_and(open>0 AND high>0 AND low>0 AND close>0 AND high>=greatest(open,close) AND low<=least(open,close) AND high>=low) AS coherent FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end GROUP BY bar_time ORDER BY bar_time"""
CLOCK = Path("data/fast_empty_template_loss_shock_clocks_2023_2026.csv.gz")
RESULT = Path("results/fast_empty_template_loss_shock_support_2026-08-09.json")
COLUMNS = (
    "candidate", "split", "anchor_height", "confirmation_height", "decision_time",
    "entry_time", "exit_time", "side", "interblock_seconds", "predecessor_weight",
    "side_return",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def ceil_5m(seconds: int) -> pd.Timestamp:
    return pd.Timestamp(((seconds + 299) // 300) * 300, unit="s", tz="UTC")


def engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env
    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def load_bars() -> pd.DataFrame:
    from sqlalchemy import text
    db = engine()
    try:
        with db.connect() as connection:
            return pd.read_sql_query(
                text(BAR_QUERY), connection,
                params={"start": (START - pd.Timedelta(hours=4)).to_pydatetime(), "end": END.to_pydatetime()},
            )
    finally:
        db.dispose()


def validate_blocks(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"height", "id", "previousblockhash", "timestamp", "mediantime", "tx_count", "size", "weight"}
    if set(frame) != required:
        raise ValueError("FETLS block schema drift")
    blocks = frame.sort_values("height", kind="mergesort").reset_index(drop=True)
    for column in ("height", "timestamp", "mediantime", "tx_count", "size", "weight"):
        blocks[column] = pd.to_numeric(blocks[column], errors="raise").astype("int64")
    if not np.array_equal(np.diff(blocks.height), np.ones(len(blocks) - 1, dtype=np.int64)):
        raise RuntimeError("block heights are not contiguous")
    if not blocks.previousblockhash.iloc[1:].reset_index(drop=True).equals(blocks.id.iloc[:-1].reset_index(drop=True)):
        raise RuntimeError("canonical previous-hash continuity failed")
    if blocks[["timestamp", "mediantime", "tx_count", "size", "weight"]].le(0).any().any():
        raise RuntimeError("nonpositive block source field")
    return blocks


def build_clock(blocks: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    blocks = validate_blocks(blocks)
    bars = bars.copy()
    for column in ("bar_time", "first_ts", "last_ts"):
        bars[column] = pd.to_datetime(bars[column], utc=True, errors="coerce")
    for column in ("bar_open", "bar_close", "source_rows", "distinct_rows"):
        bars[column] = pd.to_numeric(bars[column], errors="coerce")
    bars = bars.sort_values("bar_time", kind="mergesort").set_index("bar_time")
    prefix_time = np.maximum.accumulate(np.maximum(blocks.timestamp.to_numpy(), blocks.mediantime.to_numpy()))
    rows = []
    reserved_until = None
    for index in range(1, len(blocks) - 6):
        current = blocks.iloc[index]; previous = blocks.iloc[index - 1]
        delta = int(current.timestamp - previous.timestamp)
        if not (int(current.tx_count) == 1 and 0 < delta <= 120 and int(previous.weight) >= 3_200_000):
            continue
        confirmation = blocks.iloc[index + 6]
        decision = ceil_5m(int(prefix_time[index + 6]) + 7200)
        entry = decision + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=6)
        if entry < START or exit_time > END or (reserved_until is not None and entry < reserved_until):
            continue
        times = pd.date_range(decision - pd.Timedelta(hours=2), decision, freq="5min", inclusive="left")
        window = bars.reindex(times)
        valid = (
            np.isfinite(window[["bar_open", "bar_close", "source_rows", "distinct_rows"]]).all(axis=1)
            & window.bar_open.gt(0) & window.bar_close.gt(0)
            & window.source_rows.eq(5) & window.distinct_rows.eq(5)
            & window.coherent.fillna(False).astype(bool)
            & window.first_ts.eq(pd.Series(times, index=times))
            & window.last_ts.eq(pd.Series(times + pd.Timedelta(minutes=4), index=times))
        )
        if not bool(valid.all()):
            continue
        side_return = float(np.log(window.bar_close.iloc[-1] / window.bar_open.iloc[0]))
        if not np.isfinite(side_return) or side_return == 0:
            continue
        split = next((name for name, (left, right) in SPLITS.items() if entry >= left and exit_time <= right), None)
        if split is None:
            continue
        reserved_until = exit_time
        rows.append({
            "candidate": prereg.POLICY_ID, "split": split,
            "anchor_height": int(current.height), "confirmation_height": int(confirmation.height),
            "decision_time": decision, "entry_time": entry, "exit_time": exit_time,
            "side": int(-np.sign(side_return)), "interblock_seconds": delta,
            "predecessor_weight": int(previous.weight), "side_return": side_return,
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, float | int]:
    frame = clock[clock.split.eq(split)]
    if frame.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(frame.side.eq(1).sum()); shorts = int(frame.side.eq(-1).sum())
    months = pd.to_datetime(frame.entry_time, utc=True).dt.strftime("%Y-%m").value_counts()
    return {"events": len(frame), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(frame), "max_month_share": int(months.max()) / len(frame)}


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("FETLS preregistration drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text()); prereg.validate(registration)
    blocks = pd.read_csv(BLOCKS, compression="gzip"); bars = load_bars()
    clock = build_clock(blocks, bars); _write_gzip_csv(clock, CLOCK)
    support = {name: stats(clock, name) for name in SPLITS}; checks = {}
    for name, values in support.items():
        checks[f"{name}_minimum_events"] = values["events"] >= MINIMUM[name]
        checks[f"{name}_side_balance"] = values["minority_side_share"] >= 0.2
        checks[f"{name}_month_concentration"] = values["max_month_share"] <= 0.45
    passed = all(checks.values())
    core = {
        "protocol_version": "fetls_6_source_support_v1", "policy_id": prereg.POLICY_ID,
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "block_source": {"path": str(BLOCKS), "sha256": sha(BLOCKS), "rows": len(blocks)},
        "bar_query_sha256": hashlib.sha256(BAR_QUERY.encode()).hexdigest(),
        "completed_preentry_sources_opened": True, "execution_prices_opened": False,
        "postentry_return_or_pnl_opened": False, "gross9_rows_opened": False, "rv20_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(clock)},
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    output = {**core, "manifest_hash": chash(core)}
    RESULT.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    return output


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    result = run()
    print(json.dumps({"passed": result["support_passed"], "support": result["support"]}, indent=2))
