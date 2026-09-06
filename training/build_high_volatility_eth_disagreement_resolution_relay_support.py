"""Materialize source-only HVEDR-6 clocks."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_eth_disagreement_resolution_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
START = pd.Timestamp("2023-04-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
SOURCE_DIR = Path("data/high_volatility_eth_disagreement_resolution_relay_sources_2023_2026")
PANEL = SOURCE_DIR / "eight_hour_eth_disagreement_resolution_panel.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_eth_disagreement_resolution_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_eth_disagreement_resolution_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_eth_disagreement_resolution_relay_support_2026-08-08.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_volatility_gate",
    "no_spread_tail_gate",
    "no_direction_disagreement_gate",
    "one_boundary_stale_geometry",
    "direction_fade",
)
COLUMNS = (
    "candidate",
    "control",
    "split",
    "decision_time",
    "feature_available_time",
    "entry_time",
    "exit_time",
    "side",
    "btc_late_return","eth_late_return","relative_return_spread","absolute_relative_return_spread_rank",
    "full_variation",
    "variation_rank",
)
BTC_QUERY = """
SELECT ts,open,high,low,close
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts
"""
ETH_QUERY = BTC_QUERY.replace("symbol='BTCUSDT'", "symbol='ETHUSDT'")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def strict_prior_midrank(
    values: pd.Series, lookback: int = 270, minimum: int = 180
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    output = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        prior = history[-lookback:]
        if math.isfinite(current) and len(prior) >= minimum:
            array = np.asarray(prior)
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


def _prepare(bars: pd.DataFrame) -> pd.DataFrame:
    bars=bars.copy();bars["ts"]=pd.to_datetime(bars["ts"],utc=True)
    for column in ("open", "high", "low", "close"):
        bars[column] = pd.to_numeric(bars[column], errors="coerce")
    return bars.drop_duplicates("ts",keep=False).set_index("ts").sort_index()


def _valid_window(window:pd.DataFrame)->bool:
    finite=np.isfinite(window[["open","high","low","close"]]).all(axis=1);positive=window[["open","high","low","close"]].gt(0).all(axis=1);coherent=window.high.ge(window[["open","close"]].max(axis=1))&window.low.le(window[["open","close"]].min(axis=1))&window.high.ge(window.low);return len(window)==480 and bool((finite&positive&coherent).all())


def _eth_disagreement_resolution_panel(btc: pd.DataFrame, eth: pd.DataFrame) -> pd.DataFrame:
    btc = _prepare(btc)
    eth = _prepare(eth)
    decisions = pd.date_range(START.ceil("8h"), END, freq="8h", inclusive="left")
    rows = []
    for decision in decisions:
        expected = pd.date_range(
            decision - pd.Timedelta(hours=8), decision, freq="1min", inclusive="left"
        )
        btc_window = btc.reindex(expected)
        eth_window = eth.reindex(expected)
        valid = _valid_window(btc_window) and _valid_window(eth_window)
        if valid:
            btc_late_return = float(np.log(float(btc_window.close.iloc[-1]) / float(btc_window.open.iloc[360])))
            eth_late_return = float(np.log(float(eth_window.close.iloc[-1]) / float(eth_window.open.iloc[360])))
            relative_return_spread = eth_late_return - btc_late_return
            full_variation = float(np.log(btc_window.close.astype(float)).diff().dropna().pow(2).sum())
            valid = full_variation > 0 and math.isfinite(relative_return_spread)
        else:
            btc_late_return = eth_late_return = relative_return_spread = full_variation = float("nan")
        rows.append(
            {
                "decision_time": decision,
                "btc_source_rows": int(btc_window.notna().all(axis=1).sum()),
                "eth_source_rows": int(eth_window.notna().all(axis=1).sum()),
                "source_valid": valid,
                "btc_late_return": btc_late_return,
                "eth_late_return": eth_late_return,
                "relative_return_spread": relative_return_spread,
                "full_variation": full_variation,
            }
        )
    panel = pd.DataFrame(rows)
    panel["variation_rank"] = strict_prior_midrank(panel.full_variation)
    panel["absolute_relative_return_spread_rank"] = strict_prior_midrank(panel.relative_return_spread.abs())
    return panel


def materialize() -> dict[str, Any]:
    from sqlalchemy import text

    database = postgres_engine()
    with database.connect() as connection:
        btc = pd.read_sql_query(
            text(BTC_QUERY),
            connection,
            params={"start": START.to_pydatetime(), "end": END.to_pydatetime()},
        )
        eth = pd.read_sql_query(text(ETH_QUERY), connection, params={"start": START.to_pydatetime(), "end": END.to_pydatetime()})
    database.dispose()
    panel = _eth_disagreement_resolution_panel(btc, eth)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(panel, PANEL)
    core = {
        "protocol_version": "hvedr_6_binance_cross_asset_source_v1",
        "queries": {"btc": BTC_QUERY, "eth": ETH_QUERY},
        "table": "bars_binance",
        "symbols": ["BTCUSDT", "ETHUSDT"],
        "interval": "1m",
        "window": [START.isoformat(), END.isoformat()],
        "exact_candidate_outcomes_opened": False,
        "candidate_incidence_opened": False,
        "no_imputation": True,
        "output": {
            "path": str(PANEL),
            "sha256": sha(PANEL),
            "rows": len(panel),
            "valid_rows": int(panel.source_valid.sum()),
        },
    }
    manifest = {**core, "manifest_hash": canonical_hash(core)}
    SOURCE_MANIFEST.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def features() -> pd.DataFrame:
    frame = pd.read_csv(PANEL, compression="gzip")
    frame["decision_time"] = pd.to_datetime(frame.decision_time, utc=True)
    frame["source_valid"] = frame.source_valid.astype(str).str.lower().eq("true")
    numeric = (
        "btc_late_return","eth_late_return","relative_return_spread","absolute_relative_return_spread_rank",
        "full_variation",
        "variation_rank",
    )
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["signal_valid"] = (
        frame.source_valid
        & np.isfinite(frame[list(numeric)]).all(axis=1)
        & frame.full_variation.gt(0)
    )
    return frame


def conditions(frame: pd.DataFrame, control: str) -> tuple[pd.Series, pd.Series]:
    btc_return = frame.btc_late_return
    eth_return = frame.eth_late_return
    spread = frame.relative_return_spread
    spread_rank = frame.absolute_relative_return_spread_rank
    rank = frame.variation_rank
    if control == "one_boundary_stale_geometry":
        btc_return = btc_return.shift(1)
        eth_return = eth_return.shift(1)
        spread = spread.shift(1)
        spread_rank = spread_rank.shift(1)
        rank = rank.shift(1)
    volatility_gate = (
        pd.Series(True, index=frame.index)
        if control == "no_volatility_gate"
        else rank.ge(0.65)
    )
    tail_gate = pd.Series(True, index=frame.index) if control == "no_spread_tail_gate" else spread_rank.ge(0.8)
    disagreement_gate = pd.Series(True, index=frame.index) if control == "no_direction_disagreement_gate" else ((btc_return.gt(0) & eth_return.lt(0)) | (btc_return.lt(0) & eth_return.gt(0)))
    long = eth_return.gt(0)
    short = eth_return.lt(0)
    active = (
        frame.signal_valid
        & np.isfinite(btc_return)&np.isfinite(eth_return)&np.isfinite(spread)&np.isfinite(spread_rank)
        & np.isfinite(rank)
        & volatility_gate
        & tail_gate
        & disagreement_gate
        & (long | short)
    )
    side = pd.Series(np.where(long, 1, -1), index=frame.index)
    if control == "direction_fade":
        side = -side
    return active, side


def clock(frame: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    active, side = conditions(frame, control)
    rows = []
    for index in frame.index[active]:
        decision = pd.Timestamp(frame.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=5)
        exit_ = entry + pd.Timedelta(hours=6)
        split = next(
            (
                name
                for name, (start, end) in SPLITS.items()
                if entry >= start and exit_ <= end
            ),
            None,
        )
        if split is None:
            continue
        rows.append(
            {
                "candidate": "HVEDR-6",
                "control": control,
                "split": split,
                "decision_time": decision,
                "feature_available_time": decision,
                "entry_time": entry,
                "exit_time": exit_,
                "side": int(side.at[index]),
                "btc_late_return": float(frame.at[index, "btc_late_return"]),
                "eth_late_return": float(frame.at[index, "eth_late_return"]),
                "relative_return_spread": float(frame.at[index, "relative_return_spread"]),
                "absolute_relative_return_spread_rank": float(frame.at[index, "absolute_relative_return_spread_rank"]),
                "full_variation": float(frame.at[index, "full_variation"]),
                "variation_rank": float(frame.at[index, "variation_rank"]),
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def stats(candidate: pd.DataFrame, split: str) -> dict[str, float | int]:
    subset = candidate[candidate.split.eq(split)]
    if subset.empty:
        return {
            "events": 0,
            "longs": 0,
            "shorts": 0,
            "minority_side_share": 0.0,
            "max_month_share": 0.0,
        }
    longs = int(subset.side.eq(1).sum())
    shorts = int(subset.side.eq(-1).sum())
    months = subset.entry_time.dt.strftime("%Y-%m").value_counts()
    return {
        "events": len(subset),
        "longs": longs,
        "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(subset),
        "max_month_share": int(months.max()) / len(subset),
    }


def run() -> dict[str, Any]:
    source_manifest = materialize()
    frame = features()
    primary = clock(frame)
    controls = {name: clock(frame, name) for name in CONTROLS}
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(primary, CLOCK)
    for name, candidate in controls.items():
        _write_gzip_csv(candidate, CONTROL_DIR / f"{name}.csv.gz")
    support = {name: stats(primary, name) for name in SPLITS}
    checks = {}
    for name, values in support.items():
        checks[f"{name}_minimum_events"] = values["events"] >= MINIMUM[name]
        checks[f"{name}_side_balance"] = values["minority_side_share"] >= 0.2
        checks[f"{name}_month_concentration"] = values["max_month_share"] <= 0.45
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    passed = all(checks.values())
    core = {
        "protocol_version": "hvedr_6_source_support_v1",
        "policy_id": "HVEDR-6",
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT),
            "sha256": sha(prereg.DEFAULT_OUTPUT),
            "manifest_hash": registration["manifest_hash"],
        },
        "source_manifest": {
            "path": str(SOURCE_MANIFEST),
            "sha256": sha(SOURCE_MANIFEST),
            "manifest_hash": source_manifest["manifest_hash"],
        },
        "completed_preentry_sources_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {
            name: {
                "path": str(CONTROL_DIR / f"{name}.csv.gz"),
                "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"),
                "rows": len(candidate),
                "promotion_authorized": False,
            }
            for name, candidate in controls.items()
        },
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
