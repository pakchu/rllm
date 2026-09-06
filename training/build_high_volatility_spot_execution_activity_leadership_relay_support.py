"""Build the source-only support gate for frozen HVSEAL-8."""
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

from training import preregister_high_volatility_spot_execution_activity_leadership_relay as prereg


ENV_FILE = "/home/pakchu/rllm/.env"
START = pd.Timestamp("2023-01-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA = "f3718e4f6884cc5f972870df53d93e384f9167cd3f8ff048976e5774c2ae201c"
REGISTRATION = prereg.build()
POLICY = REGISTRATION["policy"]
SPLITS = {
    name: (pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1]))
    for name, bounds in REGISTRATION["stages"].items()
}
GATES = REGISTRATION["source_support_gates"]
CONTROLS = tuple(REGISTRATION["diagnostic_controls"]["names"])
VENUES = ("spot", "perpetual")

QUERY = """WITH source AS (
 SELECT ts,'spot'::text AS venue,open,high,low,close,number_of_trades
 FROM bars_binance_spot
 WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
 UNION ALL
 SELECT ts,'perpetual'::text AS venue,open,high,low,close,number_of_trades
 FROM bars_binance
 WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
)
SELECT date_bin('5 minutes',ts,TIMESTAMPTZ '1970-01-01 00:00:00+00') AS date,
 venue,(array_agg(open ORDER BY ts))[1] AS open,max(high) AS high,min(low) AS low,
 (array_agg(close ORDER BY ts DESC))[1] AS close,sum(number_of_trades) AS execution_count,
 count(*) AS source_rows,count(DISTINCT ts) AS distinct_rows,min(ts) AS first_ts,max(ts) AS last_ts,
 bool_and(open>0 AND high>0 AND low>0 AND close>0 AND number_of_trades>=0
          AND number_of_trades=floor(number_of_trades)
          AND high>=greatest(open,close,low) AND low<=least(open,close,high)) AS coherent
FROM source GROUP BY 1,2 ORDER BY 1,2"""

ROOT = Path("data/high_volatility_spot_execution_activity_leadership_relay_sources_2023_2026")
PANEL = ROOT / "scheduled_execution_activity_states.csv.gz"
MANIFEST = ROOT / "manifest.json"
CLOCK = Path("data/high_volatility_spot_execution_activity_leadership_relay_clocks_2023_2026.csv.gz")
SPLIT_DIR = Path("data/high_volatility_spot_execution_activity_leadership_relay_split_clocks_2023_2026")
CONTROL_DIR = Path("data/high_volatility_spot_execution_activity_leadership_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_spot_execution_activity_leadership_relay_support_2026-08-13.json")
BUILDER = Path("training/build_high_volatility_spot_execution_activity_leadership_relay_support.py")

PANEL_COLUMNS = (
    "decision_time", "feature_available_time", "source_valid",
    "spot_to_perpetual_response", "perpetual_to_spot_response",
    "leadership_margin", "leadership_rank", "perpetual_realized_variation",
    "variation_rank", "completed_perpetual_return", "direction_side", "eligible",
)
CLOCK_COLUMNS = (
    "candidate", "control", "split", "decision_time", "feature_available_time",
    "entry_time", "exit_time", "side", "spot_to_perpetual_response",
    "perpetual_to_spot_response", "leadership_margin", "leadership_rank",
    "perpetual_realized_variation", "variation_rank", "completed_perpetual_return",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def strict_prior_midrank(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(float)
    output = np.full(len(values), np.nan)
    history: list[float] = []
    for index, current in enumerate(values):
        prior = np.asarray(history[-POLICY["prior_blocks"] :], dtype=float)
        if math.isfinite(current) and len(prior) >= POLICY["minimum_prior_blocks"]:
            output[index] = (
                np.count_nonzero(prior < current)
                + 0.5 * np.count_nonzero(prior == current)
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
                text(QUERY),
                connection,
                params={"start": START.to_pydatetime(), "end": END.to_pydatetime()},
            )
    finally:
        engine.dispose()


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    expected = [
        "date", "venue", "open", "high", "low", "close", "execution_count",
        "source_rows", "distinct_rows", "first_ts", "last_ts", "coherent",
    ]
    if raw.columns.tolist() != expected:
        raise RuntimeError("HVSEAL source schema drift")
    frame = raw.copy()
    for column in ("date", "first_ts", "last_ts"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    frame["venue"] = frame["venue"].astype(str)
    for column in expected[2:9]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["date", "first_ts", "last_ts"]].isna().any().any():
        raise RuntimeError("HVSEAL invalid source timestamp")
    if frame.duplicated(["date", "venue"]).any() or not frame["venue"].isin(VENUES).all():
        raise RuntimeError("HVSEAL duplicate or unexpected source key")
    prices = frame[["open", "high", "low", "close"]]
    frame["valid"] = (
        frame["source_rows"].eq(5)
        & frame["distinct_rows"].eq(5)
        & frame["first_ts"].eq(frame["date"])
        & frame["last_ts"].eq(frame["date"] + pd.Timedelta("4m"))
        & frame["coherent"].eq(True)
        & np.isfinite(prices).all(axis=1)
        & prices.gt(0).all(axis=1)
        & frame["high"].ge(prices[["open", "close"]].max(axis=1))
        & frame["low"].le(prices[["open", "close"]].min(axis=1))
        & np.isfinite(frame["execution_count"])
        & frame["execution_count"].ge(0)
        & frame["execution_count"].eq(np.floor(frame["execution_count"]))
    )
    frame["bar_return"] = np.log(frame["close"] / frame["open"])
    return frame.sort_values(["date", "venue"], kind="mergesort")


def _positive_sample_variance(values: np.ndarray) -> bool:
    return (
        np.isfinite(values).all()
        and len(values) > 1
        and float(np.var(values, ddof=1)) > 0
    )


def block_metrics(block: pd.DataFrame) -> dict[str, Any]:
    expected_dates = pd.date_range(block["date"].min(), periods=96, freq="5min") if len(block) else []
    source_valid = (
        len(block) == 192
        and block["valid"].eq(True).all()
        and block.groupby("venue").size().reindex(VENUES).eq(96).all()
        and all(
            block.loc[block["venue"].eq(venue), "date"].reset_index(drop=True).equals(
                pd.Series(expected_dates)
            )
            for venue in VENUES
        )
    )
    if not source_valid:
        return {
            "source_valid": False,
            "spot_to_perpetual_response": math.nan,
            "perpetual_to_spot_response": math.nan,
            "leadership_margin": math.nan,
            "perpetual_realized_variation": math.nan,
            "completed_perpetual_return": math.nan,
            "direction_side": 0,
        }
    venue_frames = {
        venue: block.loc[block["venue"].eq(venue)].sort_values("date").reset_index(drop=True)
        for venue in VENUES
    }
    spot_activity = np.log1p(venue_frames["spot"]["execution_count"].to_numpy(float))
    perpetual_activity = np.log1p(
        venue_frames["perpetual"]["execution_count"].to_numpy(float)
    )
    spot_returns = venue_frames["spot"]["bar_return"].to_numpy(float)
    perpetual_returns = venue_frames["perpetual"]["bar_return"].to_numpy(float)
    vectors = (
        spot_activity[:-1], perpetual_returns[1:],
        perpetual_activity[:-1], spot_returns[1:],
    )
    if not all(_positive_sample_variance(vector) for vector in vectors):
        source_valid = False
        spot_to_perpetual = perpetual_to_spot = math.nan
    else:
        spot_to_perpetual = float(np.corrcoef(vectors[0], vectors[1])[0, 1])
        perpetual_to_spot = float(np.corrcoef(vectors[2], vectors[3])[0, 1])
    perpetual_variation = float(np.sqrt(np.square(perpetual_returns).sum()))
    perpetual = venue_frames["perpetual"]
    completed_return = float(np.log(perpetual["close"].iloc[-1] / perpetual["open"].iloc[0]))
    margin = abs(spot_to_perpetual) - abs(perpetual_to_spot)
    metrics = np.asarray(
        [spot_to_perpetual, perpetual_to_spot, margin, perpetual_variation, completed_return]
    )
    source_valid = bool(source_valid and np.isfinite(metrics).all() and perpetual_variation > 0)
    side = (
        int(np.sign(spot_to_perpetual))
        if source_valid
        and spot_to_perpetual != 0
        and completed_return != 0
        and np.sign(spot_to_perpetual) == np.sign(completed_return)
        else 0
    )
    return {
        "source_valid": source_valid,
        "spot_to_perpetual_response": spot_to_perpetual if source_valid else math.nan,
        "perpetual_to_spot_response": perpetual_to_spot if source_valid else math.nan,
        "leadership_margin": margin if source_valid and margin > 0 else math.nan,
        "perpetual_realized_variation": perpetual_variation if source_valid else math.nan,
        "completed_perpetual_return": completed_return if source_valid else math.nan,
        "direction_side": side,
    }


def build_panel(raw: pd.DataFrame) -> pd.DataFrame:
    source = prepare(raw)
    rows: list[dict[str, Any]] = []
    for decision in pd.date_range(START + pd.Timedelta("8h"), END, freq="8h", inclusive="left"):
        selected = source[
            source["date"].ge(decision - pd.Timedelta("8h"))
            & source["date"].lt(decision)
        ]
        rows.append(
            {
                "decision_time": decision,
                "feature_available_time": decision,
                **block_metrics(selected),
            }
        )
    panel = pd.DataFrame(rows)
    valid = panel["source_valid"].eq(True)
    panel["leadership_rank"] = strict_prior_midrank(
        panel["leadership_margin"].where(valid)
    )
    panel["variation_rank"] = strict_prior_midrank(
        panel["perpetual_realized_variation"].where(valid)
    )
    panel["eligible"] = (
        valid
        & panel["leadership_margin"].gt(0)
        & panel["leadership_rank"].ge(POLICY["leadership_rank_min"])
        & panel["variation_rank"].ge(POLICY["variation_rank_min"])
        & panel["direction_side"].ne(0)
    )
    return panel.loc[:, PANEL_COLUMNS]


def active_and_side(panel: pd.DataFrame, control: str = "primary") -> tuple[pd.Series, pd.Series]:
    if control not in ("primary", *CONTROLS):
        raise ValueError(f"unknown HVSEAL control: {control}")
    used = panel.copy()
    if control == "one_block_stale_geometry":
        geometry = [
            "spot_to_perpetual_response", "perpetual_to_spot_response",
            "leadership_margin", "leadership_rank", "direction_side",
        ]
        used[geometry] = used[geometry].shift(1)
    leadership_gate = (
        pd.Series(True, index=used.index)
        if control == "no_leadership_tail"
        else used["leadership_rank"].ge(POLICY["leadership_rank_min"])
    )
    margin_gate = (
        used["spot_to_perpetual_response"].abs().gt(0)
        if control == "no_reverse_channel_comparator"
        else used["leadership_margin"].gt(0)
    )
    variation_gate = (
        pd.Series(True, index=used.index)
        if control == "no_variation_gate"
        else used["variation_rank"].ge(POLICY["variation_rank_min"])
    )
    if control == "raw_spot_count_level":
        leadership_gate = used["spot_to_perpetual_response"].abs().ge(0.10)
    side = used["direction_side"].fillna(0).astype(int)
    eligible = used["source_valid"].eq(True) & margin_gate & leadership_gate & variation_gate & side.ne(0)
    decisions = pd.to_datetime(panel["decision_time"], utc=True)
    adjacent = decisions.shift(1).add(pd.Timedelta("8h")).eq(decisions)
    onset = (
        eligible
        & adjacent
        & panel["source_valid"].shift(1, fill_value=False).eq(True)
        & ~eligible.shift(1, fill_value=False)
    )
    if control == "direction_flip":
        side = -side
    elif control == "forced_long":
        side = side.where(side.eq(0), 1)
    return onset, side


def build_clock(panel: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    onset, side = active_and_side(panel, control)
    rows: list[dict[str, Any]] = []
    reserved_until: pd.Timestamp | None = None
    for index in panel.index[onset]:
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
        rows.append(
            {
                "candidate": "HVSEAL-8",
                "control": control,
                "split": split,
                "decision_time": decision,
                "feature_available_time": decision,
                "entry_time": entry,
                "exit_time": exit_time,
                "side": int(side.at[index]),
                **{
                    column: float(panel.at[index, column])
                    for column in CLOCK_COLUMNS[8:]
                },
            }
        )
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def support_stats(clock: pd.DataFrame, split: str) -> dict[str, float | int]:
    selected = clock[clock["split"].eq(split)]
    if selected.empty:
        return {
            "events": 0, "longs": 0, "shorts": 0,
            "minority_side_share": 0.0, "max_month_share": 0.0,
        }
    longs = int(selected["side"].eq(1).sum())
    shorts = int(selected["side"].eq(-1).sum())
    months = pd.to_datetime(selected["entry_time"], utc=True).dt.strftime("%Y-%m").value_counts()
    return {
        "events": len(selected),
        "longs": longs,
        "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(selected),
        "max_month_share": int(months.max()) / len(selected),
    }


def deterministic_gzip(frame: pd.DataFrame) -> bytes:
    raw = frame.to_csv(index=False, float_format="%.12g", lineterminator="\n").encode()
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=6, mtime=0) as stream:
        stream.write(raw)
    return buffer.getvalue()


def json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def immutable_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != payload:
        raise RuntimeError(f"refusing to overwrite immutable artifact {path}")
    path.write_bytes(payload)


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVSEAL preregistration drift")
    raw = load_source()
    panel = build_panel(raw)
    primary = build_clock(panel)
    controls = {name: build_clock(panel, name) for name in CONTROLS}
    immutable_write(PANEL, deterministic_gzip(panel))
    immutable_write(CLOCK, deterministic_gzip(primary))
    for name, frame in controls.items():
        immutable_write(CONTROL_DIR / f"{name}.csv.gz", deterministic_gzip(frame))
    for name in SPLITS:
        immutable_write(
            SPLIT_DIR / f"{name}.csv.gz",
            deterministic_gzip(primary[primary["split"].eq(name)]),
        )
    source_core = {
        "protocol_version": "hvseal_8_source_v1",
        "query": QUERY,
        "query_sha256": hashlib.sha256(QUERY.encode()).hexdigest(),
        "window": [START.isoformat(), END.isoformat()],
        "physical_rows": len(raw),
        "builder": {"path": str(BUILDER), "sha256": sha(BUILDER)},
        "panel": {
            "path": str(PANEL), "sha256": sha(PANEL), "rows": len(panel),
            "valid_rows": int(panel["source_valid"].sum()),
        },
        "outcomes_opened": False,
        "gross9_rows_opened": False,
        "no_imputation": True,
    }
    manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}
    immutable_write(MANIFEST, json_bytes(manifest))
    support = {name: support_stats(primary, name) for name in SPLITS}
    checks = {
        key: outcome
        for name, values in support.items()
        for key, outcome in (
            (f"{name}_minimum_events", values["events"] >= GATES["minimum_events"][name]),
            (f"{name}_side_balance", values["minority_side_share"] >= GATES["minority_side_share_min"]),
            (f"{name}_month_concentration", values["max_month_share"] <= GATES["max_month_share"]),
        )
    }
    passed = all(checks.values())
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    core = {
        "protocol_version": "hvseal_8_source_support_v1",
        "policy_id": "HVSEAL-8",
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
            name: {
                "path": str(SPLIT_DIR / f"{name}.csv.gz"),
                "sha256": sha(SPLIT_DIR / f"{name}.csv.gz"),
                "rows": int(primary["split"].eq(name).sum()),
            }
            for name in SPLITS
        },
        "controls": {
            name: {
                "path": str(CONTROL_DIR / f"{name}.csv.gz"),
                "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"),
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
    result = {**core, "manifest_hash": canonical_hash(core)}
    immutable_write(RESULT, json_bytes(result))
    return result


if __name__ == "__main__":
    report = run()
    print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
