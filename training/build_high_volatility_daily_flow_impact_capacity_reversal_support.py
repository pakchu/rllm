"""Build outcome-blind source support for frozen HVDFICR-12."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import build_high_volatility_quarter_hour_order_flow_relay_support as common
from training import preregister_high_volatility_daily_flow_impact_capacity_reversal as prereg


ENV_FILE = "/home/pakchu/rllm/.env"
START = pd.Timestamp("2023-01-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA = "e3b1b63ca127cacea6bdb81f5b134ef1ceaadfbb3a127314c3d2b62fce8d2ed9"
REGISTRATION = prereg.build()
POLICY = REGISTRATION["policy"]
SPLITS = {
    name: tuple(map(pd.Timestamp, bounds))
    for name, bounds in REGISTRATION["stages"].items()
}
GATES = REGISTRATION["source_support_gates"]
CONTROLS = tuple(REGISTRATION["diagnostic_controls"]["names"])

QUERY = """
SELECT
  date_bin('1 hour', ts, TIMESTAMPTZ '1970-01-01 00:00:00+00') AS hour_time,
  (array_agg(open ORDER BY ts))[1] AS hour_open,
  (array_agg(close ORDER BY ts DESC))[1] AS hour_close,
  sum(quote_asset_volume) AS quote_turnover,
  sum(2*taker_buy_quote-quote_asset_volume) AS signed_taker_quote,
  sum(power(ln(close/open), 2)) AS minute_squared_return,
  count(*) AS source_rows,
  count(DISTINCT ts) AS distinct_rows,
  min(ts) AS first_ts,
  max(ts) AS last_ts,
  bool_and(
    open>0 AND high>0 AND low>0 AND close>0
    AND high>=greatest(open,close,low)
    AND low<=least(open,close,high)
    AND quote_asset_volume>=0
    AND taker_buy_quote>=0
    AND taker_buy_quote<=quote_asset_volume
  ) AS coherent
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
GROUP BY 1
ORDER BY 1
"""

SOURCE_DIR = Path("data/high_volatility_daily_flow_impact_capacity_reversal_sources_2023_2026")
PANEL = SOURCE_DIR / "daily_capacity_states.csv.gz"
MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_daily_flow_impact_capacity_reversal_clocks_2023_2026.csv.gz")
SPLIT_DIR = Path("data/high_volatility_daily_flow_impact_capacity_reversal_split_clocks_2023_2026")
CONTROL_DIR = Path("data/high_volatility_daily_flow_impact_capacity_reversal_controls_2023_2026")
RESULT = Path("results/high_volatility_daily_flow_impact_capacity_reversal_support_2026-08-13.json")
BUILDER = Path(__file__).relative_to(Path.cwd())

PANEL_COLUMNS = (
    "source_day", "decision_time", "feature_available_time", "base_source_valid",
    "source_valid", "impact_beta", "impact_rank", "negative_beta_rank",
    "aggregate_flow", "realized_variation", "variation_rank", "eligible", "onset",
)
CLOCK_COLUMNS = (
    "candidate", "control", "split", "source_day", "decision_time",
    "feature_available_time", "entry_time", "exit_time", "side",
    "impact_beta", "impact_rank", "aggregate_flow", "realized_variation", "variation_rank",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return prereg.canonical_hash(value)


def prior_rank(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(float)
    output = np.full(len(values), np.nan)
    history: list[float] = []
    for index, current in enumerate(values):
        prior = np.asarray(history[-POLICY["history_days"] :], dtype=float)
        if math.isfinite(current) and len(prior) >= POLICY["minimum_history_days"]:
            output[index] = (
                np.sum(prior < current) + 0.5 * np.sum(prior == current)
            ) / len(prior)
        if math.isfinite(current):
            history.append(float(current))
    return pd.Series(output, index=series.index)


def postgres_engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(
        postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10}
    )


def load_source() -> pd.DataFrame:
    from sqlalchemy import text

    engine = postgres_engine()
    try:
        with engine.connect() as connection:
            return pd.read_sql_query(
                text(QUERY), connection, params={"start": START, "end": END}
            )
    finally:
        engine.dispose()


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    required = [
        "hour_time", "hour_open", "hour_close", "quote_turnover", "signed_taker_quote",
        "minute_squared_return", "source_rows", "distinct_rows", "first_ts", "last_ts", "coherent",
    ]
    if raw.columns.tolist() != required:
        raise RuntimeError("HVDFICR source schema drift")
    frame = raw.copy()
    for column in ("hour_time", "first_ts", "last_ts"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    numeric = (
        "hour_open", "hour_close", "quote_turnover", "signed_taker_quote",
        "minute_squared_return", "source_rows", "distinct_rows",
    )
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame.hour_time.isna().any() or frame.hour_time.duplicated().any():
        raise RuntimeError("HVDFICR invalid hourly key")
    frame["hour_valid"] = (
        np.isfinite(frame[list(numeric)]).all(axis=1)
        & frame.hour_open.gt(0)
        & frame.hour_close.gt(0)
        & frame.quote_turnover.gt(0)
        & frame.minute_squared_return.ge(0)
        & frame.source_rows.eq(60)
        & frame.distinct_rows.eq(60)
        & frame.first_ts.eq(frame.hour_time)
        & frame.last_ts.eq(frame.hour_time + pd.Timedelta(minutes=59))
        & frame.coherent.eq(True)
    )
    frame["hour_return"] = np.log(frame.hour_close / frame.hour_open)
    frame["hour_flow"] = frame.signed_taker_quote / frame.quote_turnover
    frame["hour_valid"] &= (
        np.isfinite(frame[["hour_return", "hour_flow"]]).all(axis=1)
        & frame.hour_return.ne(0)
        & frame.hour_flow.ne(0)
    )
    return frame.set_index("hour_time").sort_index()


def daily_geometry(window: pd.DataFrame) -> dict[str, Any]:
    base_valid = bool(len(window) == 24 and window.hour_valid.eq(True).all())
    if not base_valid:
        return {"base_source_valid": False, "source_valid": False}
    flow = window.hour_flow.to_numpy(float)
    returns = window.hour_return.to_numpy(float)
    denominator = float(np.square(flow).sum())
    beta = float(np.dot(flow, returns) / denominator) if denominator > 0 else math.nan
    quote = float(window.quote_turnover.sum())
    aggregate_flow = float(window.signed_taker_quote.sum() / quote) if quote > 0 else math.nan
    variation = float(math.sqrt(window.minute_squared_return.sum()))
    source_valid = bool(
        np.isfinite([beta, aggregate_flow, variation]).all()
        and beta > 0
        and aggregate_flow != 0
        and variation > 0
    )
    return {
        "base_source_valid": True,
        "source_valid": source_valid,
        "impact_beta": beta,
        "aggregate_flow": aggregate_flow,
        "realized_variation": variation,
    }


def build_panel(raw: pd.DataFrame) -> pd.DataFrame:
    hourly = prepare(raw)
    complete_index = pd.date_range(START, END, freq="1h", inclusive="left")
    hourly = hourly.reindex(complete_index)
    rows = []
    for decision in pd.date_range(START + pd.Timedelta(days=1), END, freq="1D", inclusive="left"):
        window = hourly.loc[decision - pd.Timedelta(days=1) : decision - pd.Timedelta(hours=1)]
        rows.append(
            {
                "source_day": decision - pd.Timedelta(days=1),
                "decision_time": decision,
                "feature_available_time": decision,
                **daily_geometry(window),
            }
        )
    panel = pd.DataFrame(rows)
    valid = panel.source_valid.eq(True)
    panel["impact_rank"] = prior_rank(panel.impact_beta.where(valid))
    negative = panel.base_source_valid.eq(True) & panel.impact_beta.lt(0)
    panel["negative_beta_rank"] = prior_rank((-panel.impact_beta).where(negative))
    panel["variation_rank"] = prior_rank(panel.realized_variation.where(valid))
    panel["eligible"] = (
        valid
        & panel.impact_rank.ge(POLICY["impact_rank_min"])
        & panel.variation_rank.ge(POLICY["variation_rank_min"])
    )
    panel["onset"] = common.previous_valid_onset(panel.eligible, valid)
    return panel.reindex(columns=PANEL_COLUMNS)


def active(panel: pd.DataFrame, control: str = "primary"):
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    used = panel.copy()
    valid = used.source_valid.eq(True)
    impact = used.impact_rank.ge(POLICY["impact_rank_min"])
    if control == "one_day_stale_capacity":
        used[["impact_beta", "impact_rank"]] = panel[["impact_beta", "impact_rank"]].shift(1)
        impact = used.impact_beta.gt(0) & used.impact_rank.ge(POLICY["impact_rank_min"])
    if control == "negative_beta_state":
        valid = used.base_source_valid.eq(True) & used.impact_beta.lt(0)
        impact = used.negative_beta_rank.ge(POLICY["impact_rank_min"])
    variation = used.variation_rank.ge(POLICY["variation_rank_min"])
    if control == "no_impact_tail":
        impact = used.impact_beta.gt(0)
    if control == "no_variation_gate":
        variation = pd.Series(True, index=used.index)
    eligible = valid & impact & variation
    onset = common.previous_valid_onset(eligible, valid)
    side = -np.sign(pd.to_numeric(used.aggregate_flow, errors="coerce")).fillna(0).astype(int)
    if control == "direction_flip":
        side = -side
    elif control == "forced_long":
        side = side.where(side.eq(0), 1)
    return onset & side.ne(0), side, used


def build_clock(panel: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    selected, side, used = active(panel, control)
    rows = []
    reserved_until = None
    for index in panel.index[selected]:
        decision = pd.Timestamp(panel.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=POLICY["entry_delay_minutes"])
        exit_time = entry + pd.Timedelta(hours=POLICY["hold_hours"])
        if reserved_until is not None and entry < reserved_until:
            continue
        split = next(
            (
                name for name, (start, end) in SPLITS.items()
                if entry >= start and exit_time <= end
            ),
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
                "source_day": panel.at[index, "source_day"],
                "decision_time": decision,
                "feature_available_time": used.at[index, "feature_available_time"],
                "entry_time": entry,
                "exit_time": exit_time,
                "side": int(side.at[index]),
                "impact_beta": float(used.at[index, "impact_beta"]),
                "impact_rank": float(used.at[index, "impact_rank"]),
                "aggregate_flow": float(used.at[index, "aggregate_flow"]),
                "realized_variation": float(used.at[index, "realized_variation"]),
                "variation_rank": float(used.at[index, "variation_rank"]),
            }
        )
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    selected = clock[clock.split.eq(split)]
    if selected.empty:
        return {
            "events": 0, "longs": 0, "shorts": 0,
            "minority_side_share": 0.0, "max_month_share": 0.0,
        }
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


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVDFICR preregistration drift")
    raw = load_source()
    panel = build_panel(raw)
    primary = build_clock(panel)
    controls = {name: build_clock(panel, name) for name in CONTROLS}
    splits = {name: primary[primary.split.eq(name)].copy() for name in SPLITS}
    common.immutable(PANEL, common.csv_gz(panel))
    common.immutable(CLOCK, common.csv_gz(primary))
    for name, clock in controls.items():
        common.immutable(CONTROL_DIR / f"{name}.csv.gz", common.csv_gz(clock))
    for name, clock in splits.items():
        common.immutable(SPLIT_DIR / f"{name}.csv.gz", common.csv_gz(clock))

    source_core = {
        "protocol_version": "hvdfcr_12_sources_v1",
        "query": QUERY,
        "query_sha256": hashlib.sha256(QUERY.encode()).hexdigest(),
        "table": "bars_binance",
        "symbol": "BTCUSDT",
        "interval": "1m",
        "window": [START.isoformat(), END.isoformat()],
        "physical_rows": len(raw),
        "builder": {"path": str(BUILDER), "sha256": sha(BUILDER)},
        "panel": {
            "path": str(PANEL), "sha256": sha(PANEL), "rows": len(panel),
            "valid_rows": int(panel.source_valid.sum()),
        },
        "outcomes_opened": False,
        "gross9_rows_opened": False,
        "no_imputation": True,
    }
    manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}
    common.immutable(MANIFEST, common.json_bytes(manifest))

    support = {name: stats(primary, name) for name in SPLITS}
    checks = {}
    for name, row in support.items():
        checks[f"{name}_minimum_events"] = row["events"] >= GATES["minimum_events"][name]
        checks[f"{name}_side_balance"] = row["minority_side_share"] >= GATES["minority_side_share_min"]
        checks[f"{name}_month_concentration"] = row["max_month_share"] <= GATES["max_month_share"]
    passed = all(checks.values())
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    core = {
        "protocol_version": "hvdfcr_12_source_support_v1",
        "policy_id": prereg.POLICY_ID,
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA,
            "manifest_hash": registration["manifest_hash"],
        },
        "source_manifest": {
            "path": str(MANIFEST), "sha256": sha(MANIFEST),
            "manifest_hash": manifest["manifest_hash"],
        },
        "completed_preentry_sources_opened": True,
        "candidate_incidence_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "funding_values_opened": False,
        "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "split_artifacts": {
            name: {"path": str(SPLIT_DIR / f"{name}.csv.gz"), "sha256": sha(SPLIT_DIR / f"{name}.csv.gz"), "rows": len(clock)}
            for name, clock in splits.items()
        },
        "controls": {
            name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(clock), "promotion_authorized": False}
            for name, clock in controls.items()
        },
        "support": support,
        "support_checks": checks,
        "support_passed": passed,
        "advance_to_gross9_novelty": passed,
        "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    common.immutable(RESULT, common.json_bytes(result))
    return result


if __name__ == "__main__":
    report = run()
    print(json.dumps({"passed": report["support_passed"], "support": report["support"]}))
