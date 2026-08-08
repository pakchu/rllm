"""Materialize outcome-blind source support for frozen STCSR-12."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import preregister_spot_trade_count_sponsorship_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
BUILDER = Path("training/build_spot_trade_count_sponsorship_relay_support.py")
PREREG_SHA = "f8125ea3a08d2f184f7c952ebc70b02626c424fc07673405a907ddccf4b115e7"
START = pd.Timestamp("2023-01-01T08:00:00Z")
END = pd.Timestamp("2026-08-01T08:00:00Z")
SOURCE_DIR = Path("data/spot_trade_count_sponsorship_relay_sources_2023_2026")
PANEL = SOURCE_DIR / "spot_trade_count_sponsorship_relay_preentry_features.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/spot_trade_count_sponsorship_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/spot_trade_count_sponsorship_relay_controls_2023_2026")
RESULT = Path("results/spot_trade_count_sponsorship_relay_support_2026-08-09.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_volatility_gate",
    "no_count_share_tail",
    "perpetual_count_share",
    "one_day_stale_share",
    "direction_flip",
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
    "spot_return",
    "perp_return",
    "spot_count_share",
    "spot_count_share_rank",
    "perpetual_count_share",
    "perpetual_count_share_rank",
    "perp_realized_variation",
    "variation_rank",
)
QUERY = """
SELECT ts,open,high,low,close,number_of_trades
FROM {table}
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts
"""
SPOT_QUERY = QUERY.format(table="bars_binance_spot")
PERP_QUERY = QUERY.format(table="bars_binance")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def strict_prior_midrank(
    values: pd.Series, lookback: int = 252, minimum: int = 126
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    output = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        prior = history[-lookback:]
        if math.isfinite(current) and len(prior) >= minimum:
            array = np.asarray(prior)
            output.at[index] = (
                np.count_nonzero(array < current)
                + 0.5 * np.count_nonzero(array == current)
            ) / len(array)
        if math.isfinite(current):
            history.append(float(current))
    return output


def postgres_engine():
    from sqlalchemy import create_engine

    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(
        postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10}
    )


def _prepare(bars: pd.DataFrame) -> pd.DataFrame:
    expected_columns = [
        "ts",
        "open",
        "high",
        "low",
        "close",
        "number_of_trades",
    ]
    if bars.columns.tolist() != expected_columns:
        raise RuntimeError("STCSR source schema drift")
    bars = bars.copy()
    bars["ts"] = pd.to_datetime(bars.ts, utc=True, errors="raise")
    for column in expected_columns[1:]:
        bars[column] = pd.to_numeric(bars[column], errors="coerce")
    return bars.drop_duplicates("ts", keep=False).set_index("ts").sort_index()


def _valid_window(window: pd.DataFrame) -> bool:
    prices = window[["open", "high", "low", "close"]]
    counts = window.number_of_trades
    coherent = (
        window.high.ge(window[["open", "close"]].max(axis=1))
        & window.low.le(window[["open", "close"]].min(axis=1))
        & window.high.ge(window.low)
    )
    return bool(
        len(window) == 1440
        and window.notna().all(axis=None)
        and np.isfinite(prices).all(axis=None)
        and prices.gt(0).all(axis=None)
        and coherent.all()
        and np.isfinite(counts).all()
        and counts.ge(0).all()
        and counts.eq(np.floor(counts)).all()
        and float(counts.sum()) > 0
    )


def daily_panel(perp: pd.DataFrame, spot: pd.DataFrame) -> pd.DataFrame:
    perp = _prepare(perp)
    spot = _prepare(spot)
    rows: list[dict[str, Any]] = []
    decisions = pd.date_range(
        START + pd.Timedelta(days=1), END, freq="1D", inclusive="left"
    )
    for decision in decisions:
        expected = pd.date_range(
            decision - pd.Timedelta(days=1), decision, freq="1min", inclusive="left"
        )
        perp_window = perp.reindex(expected)
        spot_window = spot.reindex(expected)
        valid = _valid_window(perp_window) and _valid_window(spot_window)
        if valid:
            spot_total = float(spot_window.number_of_trades.sum())
            perp_total = float(perp_window.number_of_trades.sum())
            combined_total = spot_total + perp_total
            valid = combined_total > 0
        if valid:
            spot_return = float(
                np.log(float(spot_window.close.iloc[-1]) / float(spot_window.open.iloc[0]))
            )
            perp_return = float(
                np.log(float(perp_window.close.iloc[-1]) / float(perp_window.open.iloc[0]))
            )
            spot_share = spot_total / combined_total
            perp_share = perp_total / combined_total
            variation = float(
                np.sqrt(
                    np.square(
                        np.log(
                            perp_window.close.astype(float).to_numpy()
                            / perp_window.open.astype(float).to_numpy()
                        )
                    ).sum()
                )
            )
        else:
            spot_total = perp_total = spot_return = perp_return = float("nan")
            spot_share = perp_share = variation = float("nan")
        rows.append(
            {
                "decision_time": decision,
                "spot_source_rows": int(spot_window.notna().all(axis=1).sum()),
                "perp_source_rows": int(perp_window.notna().all(axis=1).sum()),
                "source_valid": valid,
                "spot_trade_count": spot_total,
                "perp_trade_count": perp_total,
                "spot_return": spot_return,
                "perp_return": perp_return,
                "spot_count_share": spot_share,
                "perpetual_count_share": perp_share,
                "perp_realized_variation": variation,
            }
        )
    panel = pd.DataFrame(rows)
    panel["spot_count_share_rank"] = strict_prior_midrank(panel.spot_count_share)
    panel["perpetual_count_share_rank"] = strict_prior_midrank(
        panel.perpetual_count_share
    )
    panel["variation_rank"] = strict_prior_midrank(panel.perp_realized_variation)
    return panel


def materialize() -> dict[str, Any]:
    from sqlalchemy import text

    database = postgres_engine()
    try:
        with database.connect() as connection:
            spot = pd.read_sql_query(
                text(SPOT_QUERY),
                connection,
                params={"start": START.to_pydatetime(), "end": END.to_pydatetime()},
            )
            perp = pd.read_sql_query(
                text(PERP_QUERY),
                connection,
                params={"start": START.to_pydatetime(), "end": END.to_pydatetime()},
            )
    finally:
        database.dispose()
    panel = daily_panel(perp, spot)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(panel, PANEL)
    core = {
        "protocol_version": "stcsr_12_sources_v1",
        "queries": {"spot": SPOT_QUERY, "perpetual": PERP_QUERY},
        "tables": ["bars_binance_spot", "bars_binance"],
        "symbol": "BTCUSDT",
        "interval": "1m",
        "window": [START.isoformat(), END.isoformat()],
        "candidate_outcomes_opened": False,
        "candidate_incidence_opened": False,
        "execution_price_opened": False,
        "postentry_return_or_pnl_opened": False,
        "gross9_rows_opened": False,
        "no_imputation": True,
        "builder": {"path": str(BUILDER), "sha256": sha(BUILDER)},
        "output": {
            "path": str(PANEL),
            "sha256": sha(PANEL),
            "rows": len(panel),
            "valid_rows": int(panel.source_valid.sum()),
        },
    }
    manifest = {**core, "manifest_hash": canonical_hash(core)}
    SOURCE_MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    return manifest


def features() -> pd.DataFrame:
    frame = pd.read_csv(PANEL, compression="gzip")
    frame["decision_time"] = pd.to_datetime(frame.decision_time, utc=True)
    frame["source_valid"] = frame.source_valid.astype(str).str.lower().eq("true")
    numeric = (
        "spot_trade_count",
        "perp_trade_count",
        "spot_return",
        "perp_return",
        "spot_count_share",
        "perpetual_count_share",
        "perp_realized_variation",
        "spot_count_share_rank",
        "perpetual_count_share_rank",
        "variation_rank",
    )
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["signal_valid"] = (
        frame.source_valid
        & np.isfinite(
            frame[
                [
                    "spot_return",
                    "perp_return",
                    "spot_count_share",
                    "perpetual_count_share",
                    "perp_realized_variation",
                ]
            ]
        ).all(axis=1)
        & frame.spot_trade_count.gt(0)
        & frame.perp_trade_count.gt(0)
    )
    return frame


def conditions(frame: pd.DataFrame, control: str) -> tuple[pd.Series, pd.Series]:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    share_rank = frame.spot_count_share_rank
    if control == "one_day_stale_share":
        share_rank = share_rank.shift(1)
    if control == "perpetual_count_share":
        share_rank = frame.perpetual_count_share_rank
    share_gate = (
        pd.Series(True, index=frame.index)
        if control == "no_count_share_tail"
        else share_rank.ge(0.75)
    )
    variation_gate = (
        pd.Series(True, index=frame.index)
        if control == "no_volatility_gate"
        else frame.variation_rank.ge(0.65)
    )
    long = frame.spot_return.gt(0) & frame.perp_return.gt(0)
    short = frame.spot_return.lt(0) & frame.perp_return.lt(0)
    active = frame.signal_valid & share_gate & variation_gate & (long | short)
    side = pd.Series(np.where(long, 1, -1), index=frame.index)
    if control == "direction_flip":
        side = -side
    return active, side


def clock(frame: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    active, side = conditions(frame, control)
    rows: list[dict[str, Any]] = []
    next_allowed: pd.Timestamp | None = None
    for index in frame.index[active]:
        decision = pd.Timestamp(frame.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=12)
        if next_allowed is not None and entry < next_allowed:
            continue
        split = next(
            (
                name
                for name, (start, end) in SPLITS.items()
                if entry >= start and exit_time <= end
            ),
            None,
        )
        if split is None:
            continue
        next_allowed = exit_time
        rows.append(
            {
                "candidate": "STCSR-12",
                "control": control,
                "split": split,
                "decision_time": decision,
                "feature_available_time": decision,
                "entry_time": entry,
                "exit_time": exit_time,
                "side": int(side.at[index]),
                "spot_return": float(frame.at[index, "spot_return"]),
                "perp_return": float(frame.at[index, "perp_return"]),
                "spot_count_share": float(frame.at[index, "spot_count_share"]),
                "spot_count_share_rank": float(
                    frame.at[index, "spot_count_share_rank"]
                ),
                "perpetual_count_share": float(
                    frame.at[index, "perpetual_count_share"]
                ),
                "perpetual_count_share_rank": float(
                    frame.at[index, "perpetual_count_share_rank"]
                ),
                "perp_realized_variation": float(
                    frame.at[index, "perp_realized_variation"]
                ),
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
    months = pd.to_datetime(subset.entry_time, utc=True).dt.strftime("%Y-%m")
    return {
        "events": len(subset),
        "longs": longs,
        "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(subset),
        "max_month_share": int(months.value_counts().max()) / len(subset),
    }


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("STCSR preregistration drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    source_manifest = materialize()
    frame = features()
    primary = clock(frame)
    controls = {name: clock(frame, name) for name in CONTROLS}
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(primary, CLOCK)
    for name, candidate in controls.items():
        _write_gzip_csv(candidate, CONTROL_DIR / f"{name}.csv.gz")
    support = {name: stats(primary, name) for name in SPLITS}
    checks: dict[str, bool] = {}
    for name, values in support.items():
        checks[f"{name}_minimum_events"] = (
            values["events"] >= MINIMUM_EVENTS[name]
        )
        checks[f"{name}_side_balance"] = values["minority_side_share"] >= 0.2
        checks[f"{name}_month_concentration"] = values["max_month_share"] <= 0.45
    passed = all(checks.values())
    core = {
        "protocol_version": "stcsr_12_source_support_v1",
        "policy_id": "STCSR-12",
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT),
            "sha256": PREREG_SHA,
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
    RESULT.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    output = run()
    print(
        json.dumps(
            {"passed": output["support_passed"], "support": output["support"]},
            indent=2,
        )
    )
