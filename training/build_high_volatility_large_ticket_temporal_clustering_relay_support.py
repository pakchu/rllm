"""Materialize source-only HVLTTC-8 support clocks."""
from __future__ import annotations

import argparse
import bisect
from collections import deque
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_large_ticket_temporal_clustering_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
PREREG_SHA = "f5da0987ff2d7f8ec0081c3eff806c0b46c44ebd93fb06ec4497163c4b7565f1"
START = pd.Timestamp("2023-04-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
ROOT = Path("data/high_volatility_large_ticket_temporal_clustering_relay_sources_2023_2026")
PANEL = ROOT / "hourly_states.csv.gz"
MANIFEST = ROOT / "manifest.json"
CLOCK = Path("data/high_volatility_large_ticket_temporal_clustering_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_large_ticket_temporal_clustering_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_large_ticket_temporal_clustering_relay_support_2026-08-13.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_clustering_tail",
    "no_variation_gate",
    "turnover_hhi_only",
    "one_hour_stale_clustering",
    "direction_flip",
    "forced_long",
)
COLUMNS = (
    "candidate", "control", "split", "decision_time", "feature_available_time",
    "entry_time", "exit_time", "side", "large_ticket_clustering", "clustering_rank",
    "turnover_hhi", "turnover_hhi_rank", "execution_hhi", "realized_variation",
    "variation_rank", "block_return", "final_hour_return",
)
QUERY = """
SELECT
  date_trunc('hour', ts) AS source_hour,
  (array_agg(open ORDER BY ts))[1] AS first_open,
  (array_agg(close ORDER BY ts DESC))[1] AS last_close,
  count(*) AS source_rows,
  count(DISTINCT ts) AS distinct_timestamps,
  min(ts) AS first_ts,
  max(ts) AS last_ts,
  bool_and(
    open IS NOT NULL AND high IS NOT NULL AND low IS NOT NULL AND close IS NOT NULL
    AND quote_asset_volume IS NOT NULL AND number_of_trades IS NOT NULL
    AND open > 0 AND high > 0 AND low > 0 AND close > 0
    AND high >= open AND high >= close AND low <= open AND low <= close AND high >= low
    AND quote_asset_volume >= 0 AND number_of_trades > 0
  ) AS coherent,
  sum(quote_asset_volume) AS turnover_sum,
  sum(quote_asset_volume * quote_asset_volume) AS turnover_square_sum,
  sum(number_of_trades) AS execution_sum,
  sum(number_of_trades::numeric * number_of_trades::numeric) AS execution_square_sum,
  sum(power(ln(close / open), 2)) AS squared_return_sum
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
GROUP BY 1
ORDER BY 1
""".strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def strict_prior_midrank(
    series: pd.Series, *, lookback: int = 2160, minimum: int = 1440
) -> pd.Series:
    output = np.full(len(series), np.nan)
    queue: deque[float] = deque()
    ordered: list[float] = []
    for index, value in enumerate(pd.to_numeric(series, errors="coerce").to_numpy(float)):
        if math.isfinite(value) and len(queue) >= minimum:
            left = bisect.bisect_left(ordered, value)
            right = bisect.bisect_right(ordered, value)
            output[index] = (left + 0.5 * (right - left)) / len(ordered)
        if math.isfinite(value):
            queue.append(value)
            bisect.insort(ordered, value)
            if len(queue) > lookback:
                old = queue.popleft()
                ordered.pop(bisect.bisect_left(ordered, old))
    return pd.Series(output, index=series.index)


def postgres_engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def build_states(hourly: pd.DataFrame) -> pd.DataFrame:
    frame = hourly.copy().sort_values("source_hour").reset_index(drop=True)
    frame["source_hour"] = pd.to_datetime(frame["source_hour"], utc=True)
    numeric = (
        "first_open", "last_close", "source_rows", "distinct_timestamps", "turnover_sum",
        "turnover_square_sum", "execution_sum", "execution_square_sum", "squared_return_sum",
    )
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["first_ts"] = pd.to_datetime(frame["first_ts"], utc=True)
    frame["last_ts"] = pd.to_datetime(frame["last_ts"], utc=True)
    frame["hour_valid"] = (
        frame["source_rows"].eq(60)
        & frame["distinct_timestamps"].eq(60)
        & frame["first_ts"].eq(frame["source_hour"])
        & frame["last_ts"].eq(frame["source_hour"] + pd.Timedelta(minutes=59))
        & frame["coherent"].astype(bool)
        & np.isfinite(frame[list(numeric)]).all(axis=1)
        & frame["turnover_sum"].gt(0)
        & frame["execution_sum"].gt(0)
        & frame["squared_return_sum"].ge(0)
    )
    consecutive = frame["source_hour"].diff().eq(pd.Timedelta(hours=1))
    frame["source_valid"] = (
        frame["hour_valid"].rolling(6, min_periods=6).sum().eq(6)
        & consecutive.rolling(5, min_periods=5).sum().eq(5)
    )
    turnover = frame["turnover_sum"].rolling(6, min_periods=6).sum()
    turnover_sq = frame["turnover_square_sum"].rolling(6, min_periods=6).sum()
    executions = frame["execution_sum"].rolling(6, min_periods=6).sum()
    executions_sq = frame["execution_square_sum"].rolling(6, min_periods=6).sum()
    frame["turnover_hhi"] = turnover_sq / turnover.pow(2)
    frame["execution_hhi"] = executions_sq / executions.pow(2)
    frame["large_ticket_clustering"] = frame["turnover_hhi"] - frame["execution_hhi"]
    frame["realized_variation"] = np.sqrt(
        frame["squared_return_sum"].rolling(6, min_periods=6).sum()
    )
    frame["block_return"] = np.log(frame["last_close"] / frame["first_open"].shift(5))
    frame["final_hour_return"] = np.log(frame["last_close"] / frame["first_open"])
    frame["decision_time"] = frame["source_hour"] + pd.Timedelta(hours=1)
    finite = np.isfinite(
        frame[[
            "turnover_hhi", "execution_hhi", "large_ticket_clustering",
            "realized_variation", "block_return", "final_hour_return",
        ]]
    ).all(axis=1)
    frame["source_valid"] &= (
        finite
        & frame["large_ticket_clustering"].gt(0)
        & frame["realized_variation"].gt(0)
        & frame["block_return"].ne(0)
        & frame["final_hour_return"].ne(0)
    )
    valid = frame["source_valid"]
    frame["clustering_rank"] = strict_prior_midrank(frame["large_ticket_clustering"].where(valid))
    frame["turnover_hhi_rank"] = strict_prior_midrank(frame["turnover_hhi"].where(valid))
    frame["variation_rank"] = strict_prior_midrank(frame["realized_variation"].where(valid))
    return frame[[
        "decision_time", "source_valid", "turnover_hhi", "turnover_hhi_rank",
        "execution_hhi", "large_ticket_clustering", "clustering_rank",
        "realized_variation", "variation_rank", "block_return", "final_hour_return",
    ]]


def materialize() -> tuple[pd.DataFrame, dict[str, Any]]:
    from sqlalchemy import text

    database = postgres_engine()
    with database.connect() as connection:
        hourly = pd.read_sql_query(
            text(QUERY), connection,
            params={"start": START.to_pydatetime(), "end": END.to_pydatetime()},
        )
    database.dispose()
    states = build_states(hourly)
    ROOT.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(states, PANEL)
    core = {
        "protocol_version": "hvlttc_8_source_v1",
        "query": QUERY,
        "table": "bars_binance",
        "symbol": "BTCUSDT",
        "interval": "1m",
        "window": [START.isoformat(), END.isoformat()],
        "outcomes_opened": False,
        "candidate_incidence_opened_before_materialization": False,
        "no_imputation": True,
        "output": {
            "path": str(PANEL), "sha256": sha256(PANEL), "rows": len(states),
            "valid_rows": int(states["source_valid"].sum()),
        },
    }
    payload = {**core, "manifest_hash": canonical_hash(core)}
    MANIFEST.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return states, payload


def conditions(states: pd.DataFrame, control: str) -> tuple[pd.Series, pd.Series]:
    clustering = states["large_ticket_clustering"]
    clustering_rank = states["clustering_rank"]
    if control == "one_hour_stale_clustering":
        clustering = clustering.shift(1)
        clustering_rank = clustering_rank.shift(1)
    if control == "no_clustering_tail":
        clustering_gate = clustering.gt(0)
    elif control == "turnover_hhi_only":
        clustering_gate = states["turnover_hhi_rank"].ge(0.80)
    else:
        clustering_gate = clustering_rank.ge(0.80)
    variation_gate = (
        pd.Series(True, index=states.index)
        if control == "no_variation_gate"
        else states["variation_rank"].ge(0.65)
    )
    direction = np.sign(states["block_return"])
    confirmation = direction.eq(np.sign(states["final_hour_return"])) & direction.ne(0)
    eligible = states["source_valid"] & clustering_gate & variation_gate & confirmation
    onset = eligible & ~eligible.shift(1, fill_value=False) & states["source_valid"].shift(1, fill_value=False)
    side = direction
    if control == "direction_flip":
        side = -side
    elif control == "forced_long":
        side = pd.Series(1, index=states.index)
    return onset, side


def clock(states: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    active, side = conditions(states, control)
    rows: list[dict[str, Any]] = []
    next_allowed: pd.Timestamp | None = None
    for index in states.index[active]:
        decision = pd.Timestamp(states.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=8)
        if next_allowed is not None and entry < next_allowed:
            continue
        split = next(
            (name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end),
            None,
        )
        if split is None:
            continue
        next_allowed = exit_time
        rows.append({
            "candidate": "HVLTTC-8", "control": control, "split": split,
            "decision_time": decision, "feature_available_time": decision,
            "entry_time": entry, "exit_time": exit_time, "side": int(side.at[index]),
            "large_ticket_clustering": float(states.at[index, "large_ticket_clustering"]),
            "clustering_rank": float(states.at[index, "clustering_rank"]),
            "turnover_hhi": float(states.at[index, "turnover_hhi"]),
            "turnover_hhi_rank": float(states.at[index, "turnover_hhi_rank"]),
            "execution_hhi": float(states.at[index, "execution_hhi"]),
            "realized_variation": float(states.at[index, "realized_variation"]),
            "variation_rank": float(states.at[index, "variation_rank"]),
            "block_return": float(states.at[index, "block_return"]),
            "final_hour_return": float(states.at[index, "final_hour_return"]),
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def stats(candidate: pd.DataFrame, split: str) -> dict[str, float | int]:
    selected = candidate[candidate["split"].eq(split)]
    if selected.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(selected["side"].eq(1).sum())
    shorts = int(selected["side"].eq(-1).sum())
    months = selected["entry_time"].dt.strftime("%Y-%m").value_counts()
    return {
        "events": int(len(selected)), "longs": longs, "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(selected),
        "max_month_share": int(months.max()) / len(selected),
    }


def run() -> dict[str, Any]:
    if sha256(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVLTTC preregistration drift")
    states, source_manifest = materialize()
    primary = clock(states)
    controls = {name: clock(states, name) for name in CONTROLS}
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(primary, CLOCK)
    for name, frame in controls.items():
        _write_gzip_csv(frame, CONTROL_DIR / f"{name}.csv.gz")
    support = {name: stats(primary, name) for name in SPLITS}
    checks: dict[str, bool] = {}
    for name, values in support.items():
        checks[f"{name}_minimum_events"] = values["events"] >= MINIMUM[name]
        checks[f"{name}_side_balance"] = values["minority_side_share"] >= 0.20
        checks[f"{name}_month_concentration"] = values["max_month_share"] <= 0.45
    passed = all(checks.values())
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    core = {
        "protocol_version": "hvlttc_8_source_support_v1",
        "policy_id": "HVLTTC-8",
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT), "sha256": sha256(prereg.DEFAULT_OUTPUT),
            "manifest_hash": registration["manifest_hash"],
        },
        "source_manifest": {
            "path": str(MANIFEST), "sha256": sha256(MANIFEST),
            "manifest_hash": source_manifest["manifest_hash"],
        },
        "completed_preentry_sources_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha256(CLOCK), "rows": len(primary)},
        "controls": {
            name: {
                "path": str(CONTROL_DIR / f"{name}.csv.gz"),
                "sha256": sha256(CONTROL_DIR / f"{name}.csv.gz"),
                "rows": len(frame), "promotion_authorized": False,
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
    payload = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return payload


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    result = run()
    print(json.dumps({"passed": result["support_passed"], "support": result["support"]}, indent=2))
