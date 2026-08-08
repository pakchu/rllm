"""Materialize source-only HVSAR-6 clocks."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_settlement_absorption_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
START = pd.Timestamp("2023-04-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
SOURCE_DIR = Path("data/high_volatility_settlement_absorption_relay_sources_2023_2026")
PANEL = SOURCE_DIR / "eight_hour_absorption_panel.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_settlement_absorption_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_settlement_absorption_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_settlement_absorption_relay_support_2026-08-08.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_volatility_gate",
    "no_volume_absorption_gate",
    "one_boundary_stale_geometry",
    "follow_early_shock",
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
    "early_open","early_close","late_open","late_close","early_return","late_return","block_quote_volume","late_quote_volume","late_quote_volume_share",
    "full_variation",
    "variation_rank",
)
QUERY = """
SELECT ts,open,high,low,close,volume,quote_asset_volume
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts
"""


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def strict_prior_midrank(
    values: pd.Series, lookback: int = 180, minimum: int = 120
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


def _absorption_panel(bars: pd.DataFrame) -> pd.DataFrame:
    bars = bars.copy()
    bars["ts"] = pd.to_datetime(bars["ts"], utc=True)
    for column in ("open", "high", "low", "close", "volume", "quote_asset_volume"):
        bars[column] = pd.to_numeric(bars[column], errors="coerce")
    bars = bars.drop_duplicates("ts", keep=False).set_index("ts").sort_index()
    decisions = pd.date_range(START.ceil("8h"), END, freq="8h", inclusive="left")
    rows = []
    for decision in decisions:
        expected = pd.date_range(
            decision - pd.Timedelta(hours=8), decision, freq="1min", inclusive="left"
        )
        window = bars.reindex(expected)
        finite = np.isfinite(window[["open", "high", "low", "close"]]).all(axis=1)
        positive = window[["open", "high", "low", "close"]].gt(0).all(axis=1)
        coherent = (
            window["high"].ge(window[["open", "close"]].max(axis=1))
            & window["low"].le(window[["open", "close"]].min(axis=1))
            & window["high"].ge(window["low"])
        )
        volume_valid = (
            np.isfinite(window[["volume", "quote_asset_volume"]]).all(axis=1)
            & window[["volume", "quote_asset_volume"]].ge(0).all(axis=1)
        )
        valid = len(window) == 480 and bool((finite & positive & coherent & volume_valid).all()) and float(window.quote_asset_volume.sum()) > 0
        if valid:
            early=window.iloc[:360];late=window.iloc[360:];early_open=float(early.open.iloc[0]);early_close=float(early.close.iloc[-1]);late_open=float(late.open.iloc[0]);late_close=float(late.close.iloc[-1]);early_return=float(np.log(early_close/early_open));late_return=float(np.log(late_close/late_open));block_quote_volume=float(window.quote_asset_volume.sum());late_quote_volume=float(late.quote_asset_volume.sum());late_quote_volume_share=late_quote_volume/block_quote_volume;full_variation=float(np.log(window.close.astype(float)).diff().dropna().pow(2).sum());valid=full_variation>0
        else:
            early_open=early_close=late_open=late_close=early_return=late_return=block_quote_volume=late_quote_volume=late_quote_volume_share=full_variation=float("nan")
        rows.append(
            {
                "decision_time": decision,
                "source_rows": int(window.notna().all(axis=1).sum()),
                "source_valid": valid,
                "early_open":early_open,"early_close":early_close,"late_open":late_open,"late_close":late_close,"early_return":early_return,"late_return":late_return,"block_quote_volume":block_quote_volume,"late_quote_volume":late_quote_volume,"late_quote_volume_share":late_quote_volume_share,"full_variation":full_variation,
            }
        )
    panel = pd.DataFrame(rows)
    panel["variation_rank"] = strict_prior_midrank(panel.full_variation)
    return panel


def materialize() -> dict[str, Any]:
    from sqlalchemy import text

    database = postgres_engine()
    with database.connect() as connection:
        bars = pd.read_sql_query(
            text(QUERY),
            connection,
            params={"start": START.to_pydatetime(), "end": END.to_pydatetime()},
        )
    database.dispose()
    panel = _absorption_panel(bars)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(panel, PANEL)
    core = {
        "protocol_version": "hvsar_6_btc_source_v1",
        "query": QUERY,
        "table": "bars_binance",
        "symbol": "BTCUSDT",
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
        "early_open","early_close","late_open","late_close","early_return","late_return","block_quote_volume","late_quote_volume","late_quote_volume_share",
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
    early=frame.early_return;late=frame.late_return;share=frame.late_quote_volume_share
    rank = frame.variation_rank
    if control == "one_boundary_stale_geometry":
        early=early.shift(1);late=late.shift(1);share=share.shift(1)
        rank = rank.shift(1)
    volatility_gate = (
        pd.Series(True, index=frame.index)
        if control == "no_volatility_gate"
        else rank.ge(0.65)
    )
    volume_gate=pd.Series(True,index=frame.index) if control=="no_volume_absorption_gate" else share.ge(.3);long=early.lt(0)&late.gt(0)&late.abs().le(.5*early.abs());short=early.gt(0)&late.lt(0)&late.abs().le(.5*early.abs())
    active = (
        frame.signal_valid
        & np.isfinite(early)&np.isfinite(late)&np.isfinite(share)
        & np.isfinite(rank)
        & volatility_gate
        & volume_gate
        & (long | short)
    )
    side = pd.Series(np.where(long, 1, -1), index=frame.index)
    if control == "follow_early_shock":
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
                "candidate": "HVSAR-6",
                "control": control,
                "split": split,
                "decision_time": decision,
                "feature_available_time": decision,
                "entry_time": entry,
                "exit_time": exit_,
                "side": int(side.at[index]),
                "early_open":float(frame.at[index,"early_open"]),"early_close":float(frame.at[index,"early_close"]),"late_open":float(frame.at[index,"late_open"]),"late_close":float(frame.at[index,"late_close"]),"early_return":float(frame.at[index,"early_return"]),"late_return":float(frame.at[index,"late_return"]),"block_quote_volume":float(frame.at[index,"block_quote_volume"]),"late_quote_volume":float(frame.at[index,"late_quote_volume"]),"late_quote_volume_share":float(frame.at[index,"late_quote_volume_share"]),
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
        "protocol_version": "hvsar_6_source_support_v1",
        "policy_id": "HVSAR-6",
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
