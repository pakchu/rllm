"""Build source-only HVLIR-8 clocks without opening post-entry outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_liquidity_impact_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
START = pd.Timestamp("2023-04-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
SOURCE_DIR = Path("data/high_volatility_liquidity_impact_relay_sources_2023_2026")
PANEL = SOURCE_DIR / "hourly_liquidity_impact_panel.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_liquidity_impact_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_liquidity_impact_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_liquidity_impact_relay_support_2026-08-09.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_volatility_gate",
    "raw_absolute_return_rank",
    "one_hour_stale_impact",
    "direction_flip",
)
QUERY = """
SELECT ts,open,high,low,close,quote_asset_volume
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts
"""
CLOCK_COLUMNS = (
    "candidate", "control", "split", "decision_time", "feature_available_time",
    "entry_time", "exit_time", "side", "hour_return", "hour_quote_turnover",
    "turnover_baseline", "normalized_impact", "impact_rank", "variation_24h",
    "variation_rank",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def strict_prior_midrank(
    values: pd.Series, lookback: int = 2160, minimum: int = 1440
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    output = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        prior = history[-lookback:]
        if math.isfinite(current) and len(prior) >= minimum:
            array = np.asarray(prior, dtype=float)
            output.at[index] = (
                np.sum(array < current) + 0.5 * np.sum(array == current)
            ) / len(array)
        if math.isfinite(current):
            history.append(current)
    return output


def postgres_engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(
        postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10}
    )


def build_panel(bars: pd.DataFrame) -> pd.DataFrame:
    frame = bars.copy()
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    numeric = ("open", "high", "low", "close", "quote_asset_volume")
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.drop_duplicates("ts", keep=False).set_index("ts").sort_index()
    full_grid = pd.date_range(START, END, freq="1min", inclusive="left")
    frame = frame.reindex(full_grid)
    finite = np.isfinite(frame[list(numeric)]).all(axis=1)
    positive = frame[["open", "high", "low", "close"]].gt(0).all(axis=1)
    coherent = (
        frame["high"].ge(frame[["open", "close"]].max(axis=1))
        & frame["low"].le(frame[["open", "close"]].min(axis=1))
        & frame["high"].ge(frame["low"])
    )
    volume_valid = frame["quote_asset_volume"].ge(0)
    frame["minute_valid"] = finite & positive & coherent & volume_valid
    frame["minute_log_return"] = np.log(frame["close"]).diff()
    group = frame.groupby(frame.index.floor("h"), sort=True)
    hourly = pd.DataFrame(
        {
            "source_rows": group["minute_valid"].size(),
            "valid_rows": group["minute_valid"].sum(),
            "hour_open": group["open"].first(),
            "hour_close": group["close"].last(),
            "hour_quote_turnover": group["quote_asset_volume"].sum(min_count=60),
            "hour_variation": group["minute_log_return"].apply(
                lambda x: float(x.pow(2).sum()) if x.notna().sum() == 60 else np.nan
            ),
        }
    )
    hourly.index.name = "hour_start"
    hourly["source_valid"] = (
        hourly["source_rows"].eq(60)
        & hourly["valid_rows"].eq(60)
        & hourly["hour_quote_turnover"].gt(0)
        & np.isfinite(hourly[["hour_open", "hour_close", "hour_quote_turnover", "hour_variation"]]).all(axis=1)
    )
    hourly["decision_time"] = hourly.index + pd.Timedelta(hours=1)
    hourly["hour_return"] = np.log(hourly["hour_close"] / hourly["hour_open"])
    turnover = hourly["hour_quote_turnover"].where(hourly["source_valid"])
    hourly["turnover_baseline"] = turnover.shift(1).rolling(168, min_periods=120).median()
    hourly["normalized_impact"] = (
        hourly["hour_return"].abs()
        / (hourly["hour_quote_turnover"] / hourly["turnover_baseline"])
    ).where(hourly["source_valid"] & hourly["hour_return"].ne(0))
    hourly["variation_24h"] = (
        hourly["hour_variation"].where(hourly["source_valid"]).rolling(24, min_periods=24).sum()
    )
    hourly["impact_rank"] = strict_prior_midrank(hourly["normalized_impact"])
    hourly["raw_return_rank"] = strict_prior_midrank(hourly["hour_return"].abs().where(hourly["source_valid"] & hourly["hour_return"].ne(0)))
    hourly["variation_rank"] = strict_prior_midrank(hourly["variation_24h"])
    hourly["signal_valid"] = (
        hourly["source_valid"]
        & np.isfinite(hourly[["hour_return", "turnover_baseline", "normalized_impact", "impact_rank", "raw_return_rank", "variation_24h", "variation_rank"]]).all(axis=1)
        & hourly["hour_return"].ne(0)
    )
    return hourly.reset_index()


def conditions(frame: pd.DataFrame, control: str = "primary") -> tuple[pd.Series, pd.Series]:
    impact_rank = frame["impact_rank"]
    variation_rank = frame["variation_rank"]
    direction = frame["hour_return"]
    if control == "raw_absolute_return_rank":
        impact_rank = frame["raw_return_rank"]
    elif control == "one_hour_stale_impact":
        impact_rank = impact_rank.shift(1)
        variation_rank = variation_rank.shift(1)
        direction = direction.shift(1)
    crossing = impact_rank.ge(0.80) & impact_rank.shift(1).lt(0.80)
    volatility = (
        pd.Series(True, index=frame.index)
        if control == "no_volatility_gate"
        else variation_rank.ge(0.65)
    )
    active = (
        frame["signal_valid"]
        & np.isfinite(impact_rank)
        & np.isfinite(variation_rank)
        & np.isfinite(direction)
        & direction.ne(0)
        & crossing
        & volatility
    )
    side = pd.Series(np.where(direction.gt(0), 1, -1), index=frame.index)
    if control == "direction_flip":
        side = -side
    return active, side


def clock(frame: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    active, side = conditions(frame, control)
    rows: list[dict[str, Any]] = []
    reserved_until: pd.Timestamp | None = None
    for index in frame.index[active]:
        decision = pd.Timestamp(frame.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=5)
        exit_ = entry + pd.Timedelta(hours=8)
        if reserved_until is not None and entry < reserved_until:
            continue
        split = next(
            (name for name, (start, end) in SPLITS.items() if entry >= start and exit_ <= end),
            None,
        )
        if split is None:
            continue
        reserved_until = exit_
        rows.append(
            {
                "candidate": "HVLIR-8",
                "control": control,
                "split": split,
                "decision_time": decision,
                "feature_available_time": decision,
                "entry_time": entry,
                "exit_time": exit_,
                "side": int(side.at[index]),
                "hour_return": float(frame.at[index, "hour_return"]),
                "hour_quote_turnover": float(frame.at[index, "hour_quote_turnover"]),
                "turnover_baseline": float(frame.at[index, "turnover_baseline"]),
                "normalized_impact": float(frame.at[index, "normalized_impact"]),
                "impact_rank": float(frame.at[index, "impact_rank"]),
                "variation_24h": float(frame.at[index, "variation_24h"]),
                "variation_rank": float(frame.at[index, "variation_rank"]),
            }
        )
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def stats(candidate: pd.DataFrame, split: str) -> dict[str, float | int]:
    subset = candidate[candidate["split"].eq(split)]
    if subset.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(subset["side"].eq(1).sum())
    shorts = int(subset["side"].eq(-1).sum())
    months = pd.to_datetime(subset["entry_time"], utc=True).dt.strftime("%Y-%m").value_counts()
    return {
        "events": int(len(subset)),
        "longs": longs,
        "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(subset),
        "max_month_share": int(months.max()) / len(subset),
    }


def run() -> dict[str, Any]:
    from sqlalchemy import text

    engine = postgres_engine()
    with engine.connect() as connection:
        bars = pd.read_sql_query(text(QUERY), connection, params={"start": START.to_pydatetime(), "end": END.to_pydatetime()})
    engine.dispose()
    panel = build_panel(bars)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(panel, PANEL)
    source_core = {
        "protocol_version": "hvlir_8_source_v1",
        "query": QUERY,
        "table": "bars_binance",
        "symbol": "BTCUSDT",
        "interval": "1m",
        "window": [START.isoformat(), END.isoformat()],
        "postentry_return_pnl_opened": False,
        "gross9_rows_opened": False,
        "output": {"path": str(PANEL), "sha256": sha(PANEL), "rows": len(panel), "valid_rows": int(panel["source_valid"].sum())},
    }
    source_manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}
    SOURCE_MANIFEST.write_text(json.dumps(source_manifest, indent=2, allow_nan=False) + "\n")
    primary = clock(panel)
    controls = {name: clock(panel, name) for name in CONTROLS}
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(primary, CLOCK)
    for name, candidate in controls.items():
        _write_gzip_csv(candidate, CONTROL_DIR / f"{name}.csv.gz")
    support = {name: stats(primary, name) for name in SPLITS}
    checks: dict[str, bool] = {}
    for name, values in support.items():
        checks[f"{name}_minimum_events"] = values["events"] >= MINIMUM[name]
        checks[f"{name}_side_balance"] = values["minority_side_share"] >= 0.20
        checks[f"{name}_month_concentration"] = values["max_month_share"] <= 0.45
    passed = all(checks.values())
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    core = {
        "protocol_version": "hvlir_8_source_support_v1",
        "policy_id": "HVLIR-8",
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": sha(prereg.DEFAULT_OUTPUT), "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(SOURCE_MANIFEST), "sha256": sha(SOURCE_MANIFEST), "manifest_hash": source_manifest["manifest_hash"]},
        "completed_preentry_sources_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(candidate), "promotion_authorized": False} for name, candidate in controls.items()},
        "support": support,
        "support_checks": checks,
        "support_passed": passed,
        "advance_to_gross9_novelty": passed,
        "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    output = run()
    print(json.dumps({"passed": output["support_passed"], "support": output["support"]}, indent=2))
