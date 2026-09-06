"""Deterministic outcome-blind source support for HVCVFR-24."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import build_high_volatility_quarter_hour_order_flow_relay_support as common
from training import preregister_high_volatility_causal_vixfix_reentry_reversal as prereg

ENV_FILE = "/home/pakchu/rllm/.env"
START = pd.Timestamp("2020-01-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA = "a93d677206bd2e2a9d87808bf0434934bedfdb9ae59dcbee19f724d5680fa07e"
REGISTRATION = prereg.build()
POLICY = REGISTRATION["policy"]
STAGES = {key: tuple(map(pd.Timestamp, value)) for key, value in REGISTRATION["stages"].items()}
GATES = REGISTRATION["source_support_gates"]

BAR_QUERY = """SELECT date_bin('1 day',ts,TIMESTAMPTZ '1970-01-01 00:00:00+00')+INTERVAL '1 day' AS decision_time,(array_agg(close ORDER BY ts DESC))[1] AS last_close,max(high) AS auction_high,min(low) AS auction_low,sum(power(ln(close/open),2)) AS minute_squared_return,count(*) AS source_rows,count(DISTINCT ts) AS distinct_rows,min(ts) AS first_ts,max(ts) AS last_ts,bool_and(open>0 AND high>0 AND low>0 AND close>0 AND high>=greatest(open,close,low) AND low<=least(open,close,high)) AS coherent FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end GROUP BY 1 ORDER BY 1"""

ROOT = Path("data/high_volatility_causal_vixfix_reentry_reversal_sources_2020_2026")
PANEL = ROOT / "settlement_states.csv.gz"
MANIFEST = ROOT / "manifest.json"
CLOCK = Path("data/high_volatility_causal_vixfix_reentry_reversal_clocks_2020_2026.csv.gz")
SPLIT_DIR = Path("data/high_volatility_causal_vixfix_reentry_reversal_split_clocks_2020_2026")
RESULT = Path("results/high_volatility_causal_vixfix_reentry_reversal_support_2026-08-16.json")
BUILDER = Path(__file__).relative_to(Path.cwd())
PANEL_COLUMNS = (
    "decision_time", "feature_available_time", "source_valid", "last_close",
    "prior_highest_close", "prior_lowest_close", "downside_displacement",
    "upside_displacement", "dominant_displacement", "reversal_side",
    "displacement_rank", "previous_displacement_rank", "previous_reversal_side",
    "realized_variation", "variation_rank", "eligible",
)
CLOCK_COLUMNS = (
    "candidate", "control", "split", "decision_time", "feature_available_time",
    "entry_time", "exit_time", "side", "dominant_displacement",
    "displacement_rank", "realized_variation", "variation_rank",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def causal_midrank(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(float)
    output = np.full(len(values), np.nan)
    history: list[float] = []
    for index, value in enumerate(values):
        prior = np.asarray(history[-POLICY["history_decisions"] :], dtype=float)
        if math.isfinite(value) and len(prior) >= POLICY["minimum_history_decisions"]:
            output[index] = float(
                (np.sum(prior < value) + 0.5 * np.sum(prior == value)) / len(prior)
            )
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
            bars = pd.read_sql_query(
                text(BAR_QUERY), connection, params={"start": START, "end": END}
            )
    finally:
        database.dispose()
    return bars


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    bars = raw
    expected_bars = [
        "decision_time", "last_close", "auction_high", "auction_low", "minute_squared_return",
        "source_rows", "distinct_rows", "first_ts", "last_ts", "coherent",
    ]
    if bars.columns.tolist() != expected_bars:
        raise RuntimeError("HVCVFR-24 bar schema drift")

    bars = bars.copy()
    for column in ("decision_time", "first_ts", "last_ts"):
        bars[column] = pd.to_datetime(bars[column], utc=True, errors="raise")
    numeric = ("last_close", "auction_high", "auction_low", "minute_squared_return", "source_rows", "distinct_rows")
    for column in numeric:
        bars[column] = pd.to_numeric(bars[column], errors="raise")
    if bars["decision_time"].duplicated().any():
        raise RuntimeError("HVCVFR-24 duplicate daily bar")
    start = bars["decision_time"] - pd.Timedelta("24h")
    bars["bar_valid"] = (
        np.isfinite(bars[list(numeric)]).all(axis=1)
        & bars["last_close"].gt(0)
        & bars["minute_squared_return"].gt(0)
        & bars["source_rows"].eq(POLICY["current_auction_minutes"])
        & bars["distinct_rows"].eq(POLICY["current_auction_minutes"])
        & bars["first_ts"].eq(start)
        & bars["last_ts"].eq(bars["decision_time"] - pd.Timedelta("1m"))
        & bars["coherent"].eq(True)
    )
    bars = bars.sort_values("decision_time").reset_index(drop=True)
    periods = POLICY["prior_close_periods"]
    bars["prior_highest_close"] = bars["last_close"].shift(1).rolling(periods, min_periods=periods).max()
    bars["prior_lowest_close"] = bars["last_close"].shift(1).rolling(periods, min_periods=periods).min()
    bars["prior_close_valid"] = (
        bars["bar_valid"].shift(1).rolling(periods, min_periods=periods).sum().eq(periods)
        & bars["decision_time"].shift(periods).eq(
            bars["decision_time"] - pd.Timedelta(hours=24 * periods)
        )
    )
    variation_periods = POLICY["variation_minutes"] // POLICY["current_auction_minutes"]
    variation_sum = bars["minute_squared_return"].rolling(variation_periods, min_periods=variation_periods).sum()
    bars["variation_valid"] = (
        bars["bar_valid"].rolling(variation_periods, min_periods=variation_periods).sum().eq(variation_periods)
        & bars["decision_time"].shift(variation_periods - 1).eq(
            bars["decision_time"] - pd.Timedelta(hours=24 * (variation_periods - 1))
        )
    )
    bars["realized_variation"] = np.sqrt(variation_sum)
    bars["downside_displacement"] = (
        bars["prior_highest_close"] - bars["auction_low"]
    ) / bars["prior_highest_close"]
    bars["upside_displacement"] = (
        bars["auction_high"] - bars["prior_lowest_close"]
    ) / bars["prior_lowest_close"]
    bars["reversal_side"] = np.select(
        [
            bars["downside_displacement"].gt(0)
            & bars["downside_displacement"].gt(bars["upside_displacement"]),
            bars["upside_displacement"].gt(0)
            & bars["upside_displacement"].gt(bars["downside_displacement"]),
        ],
        [1, -1],
        default=0,
    )
    bars["dominant_displacement"] = bars[["downside_displacement", "upside_displacement"]].max(axis=1)
    return bars


def build_panel(raw: pd.DataFrame) -> pd.DataFrame:
    panel = prepare(raw)
    panel["source_valid"] = (
        panel["bar_valid"].eq(True)
        & panel["prior_close_valid"].eq(True)
        & panel["variation_valid"].eq(True)
        & np.isfinite(
            panel[["last_close", "prior_highest_close", "prior_lowest_close", "downside_displacement", "upside_displacement", "dominant_displacement", "realized_variation"]]
        ).all(axis=1)
        & panel["reversal_side"].ne(0)
        & panel["dominant_displacement"].gt(0)
        & panel["realized_variation"].gt(0)
    )
    valid = panel["source_valid"].eq(True)
    panel["displacement_rank"] = causal_midrank(panel["dominant_displacement"].where(valid))
    panel["variation_rank"] = causal_midrank(panel["realized_variation"].where(valid))
    panel["previous_source_valid"] = panel["source_valid"].shift(1, fill_value=False)
    panel["previous_displacement_rank"] = panel["displacement_rank"].shift(1)
    panel["previous_reversal_side"] = panel["reversal_side"].shift(1)
    exact_previous_day = panel["decision_time"].sub(panel["decision_time"].shift(1)).eq(pd.Timedelta("1d"))
    panel["eligible"] = (
        valid
        & panel["previous_source_valid"]
        & exact_previous_day
        & panel["previous_displacement_rank"].ge(POLICY["displacement_tail_min"])
        & panel["displacement_rank"].lt(POLICY["displacement_tail_min"])
        & panel["reversal_side"].eq(panel["previous_reversal_side"])
        & panel["variation_rank"].ge(POLICY["variation_rank_min"])
    )
    panel["feature_available_time"] = panel["decision_time"]
    return panel.loc[:, PANEL_COLUMNS]


def stage_for(entry: pd.Timestamp, exit_: pd.Timestamp) -> str | None:
    return next(
        (name for name, (start, end) in STAGES.items() if start <= entry and exit_ <= end),
        None,
    )


def build_clock(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    reserved_until: pd.Timestamp | None = None
    for row in panel.loc[panel["eligible"]].itertuples(index=False):
        decision = pd.Timestamp(row.decision_time)
        entry = decision + pd.Timedelta(minutes=POLICY["entry_delay_minutes"])
        exit_ = entry + pd.Timedelta(hours=POLICY["hold_hours"])
        if reserved_until is not None and entry < reserved_until:
            continue
        split = stage_for(entry, exit_)
        if split is None:
            continue
        side = int(row.reversal_side)
        if side not in (-1, 1) or row.feature_available_time > entry:
            raise RuntimeError("HVCVFR-24 side or availability drift")
        reserved_until = exit_
        rows.append({
            "candidate": prereg.POLICY_ID, "control": "primary", "split": split,
            "decision_time": decision, "feature_available_time": row.feature_available_time,
            "entry_time": entry, "exit_time": exit_, "side": side,
            "dominant_displacement": float(row.dominant_displacement),
            "displacement_rank": float(row.displacement_rank),
            "realized_variation": float(row.realized_variation),
            "variation_rank": float(row.variation_rank),
        })
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def support_stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    subset = clock.loc[clock["split"].eq(split)]
    if subset.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(subset["side"].eq(1).sum()); shorts = int(subset["side"].eq(-1).sum())
    months = pd.to_datetime(subset["entry_time"], utc=True).dt.strftime("%Y-%m").value_counts()
    return {"events": len(subset), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(subset), "max_month_share": int(months.max()) / len(subset)}


def run() -> dict[str, Any]:
    if sha256_file(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVCVFR-24 preregistration hash drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    raw = load_source()
    panel = build_panel(raw)
    clock = build_clock(panel)
    split_clocks = {name: clock.loc[clock["split"].eq(name)].copy() for name in STAGES}
    common.immutable(PANEL, common.csv_gz(panel))
    common.immutable(CLOCK, common.csv_gz(clock))
    for name, frame in split_clocks.items():
        common.immutable(SPLIT_DIR / f"{name}.csv.gz", common.csv_gz(frame))
    source_core = {
        "protocol_version": "hvcvfr_24_sources_v1",
        "queries": {"bars": BAR_QUERY},
        "query_sha256": {"bars": hashlib.sha256(BAR_QUERY.encode()).hexdigest()},
        "tables": ["bars_binance"], "symbol": "BTCUSDT",
        "window": [START.isoformat(), END.isoformat()],
        "physical_rows": {"bars": len(raw)},
        "builder": {"path": str(BUILDER), "sha256": sha256_file(BUILDER)},
        "panel": {"path": str(PANEL), "sha256": sha256_file(PANEL), "rows": len(panel), "valid_rows": int(panel["source_valid"].sum())},
        "outcomes_opened": False, "execution_prices_opened": False,
        "held_interval_funding_opened": False, "gross9_rows_opened": False,
        "no_imputation": True,
    }
    manifest = {**source_core, "manifest_hash": prereg.canonical_hash(source_core)}
    common.immutable(MANIFEST, common.json_bytes(manifest))
    support = {name: support_stats(clock, name) for name in STAGES}
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
    core = {
        "protocol_version": "hvcvfr_24_source_support_v1", "policy_id": prereg.POLICY_ID,
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(MANIFEST), "sha256": sha256_file(MANIFEST), "manifest_hash": manifest["manifest_hash"]},
        "completed_preentry_sources_opened": True, "candidate_incidence_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "held_interval_funding_values_opened": False, "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha256_file(CLOCK), "rows": len(clock)},
        "split_artifacts": {name: {"path": str(SPLIT_DIR / f"{name}.csv.gz"), "sha256": sha256_file(SPLIT_DIR / f"{name}.csv.gz"), "rows": len(frame)} for name, frame in split_clocks.items()},
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False,
        "decision": "pass_to_gross9_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": prereg.canonical_hash(core)}
    common.immutable(RESULT, common.json_bytes(result))
    return result


if __name__ == "__main__":
    report = run()
    print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
