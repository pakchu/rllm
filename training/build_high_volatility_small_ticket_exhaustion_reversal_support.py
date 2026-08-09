"""Build source-only HVSTER-8 clocks before opening outcomes or Gross9."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_small_ticket_exhaustion_reversal as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


PREREG_SHA = "93a1e32146baed818affc7b3274a671c2b0158acc780f77404af24592dec8359"
MARKET = Path("data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz")
ENV_FILE = Path("/home/pakchu/rllm/.env")
EXTENSION_START = pd.Timestamp("2026-05-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
CLOCK = Path("data/high_volatility_small_ticket_exhaustion_reversal_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_small_ticket_exhaustion_reversal_controls_2023_2026")
SNAPSHOT = Path(
    "data/high_volatility_small_ticket_exhaustion_reversal_sources_2023_2026/"
    "small_ticket_scores.csv.gz"
)
RESULT = Path("results/high_volatility_small_ticket_exhaustion_reversal_support_2026-08-09.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), END),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_volatility_gate",
    "count_tail_only",
    "ticket_tail_only",
    "one_block_stale_participation",
    "direction_flip",
)
CLOCK_COLUMNS = (
    "candidate",
    "control",
    "split",
    "decision_time",
    "feature_available_time",
    "entry_time",
    "exit_time",
    "side",
)
LIVE_QUERY = """
SELECT ts, open, high, low, close, quote_asset_volume, number_of_trades
FROM bars_binance
WHERE symbol = 'BTCUSDT'
  AND interval = '1m'
  AND ts >= :start
  AND ts < :end
ORDER BY ts
"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _postgres_engine():
    from preprocessing.live_db_features import sqlalchemy_engine_from_env

    return sqlalchemy_engine_from_env(ENV_FILE)


def _five_minute_extension(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    frame["date"] = pd.to_datetime(frame.pop("ts"), utc=True)
    for column in ("open", "high", "low", "close", "quote_asset_volume", "number_of_trades"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame.date.duplicated().any():
        raise RuntimeError("HVSTER live one-minute source contains duplicate timestamps")
    frame = frame.sort_values("date").set_index("date")
    expected = pd.date_range(EXTENSION_START, END, freq="1min", inclusive="left")
    if not frame.index.equals(expected):
        raise RuntimeError("HVSTER live one-minute source is not an exact consecutive grid")
    prices = frame[["open", "high", "low", "close"]]
    coherent = (
        np.isfinite(prices).all(axis=1)
        & prices.gt(0).all(axis=1)
        & frame.high.ge(frame[["open", "close"]].max(axis=1))
        & frame.low.le(frame[["open", "close"]].min(axis=1))
        & frame.high.ge(frame.low)
    )
    quote_valid = np.isfinite(frame.quote_asset_volume) & frame.quote_asset_volume.ge(0)
    count_valid = np.isfinite(frame.number_of_trades) & frame.number_of_trades.ge(0) & frame.number_of_trades.eq(np.floor(frame.number_of_trades))
    if not bool((coherent & quote_valid & count_valid).all()):
        raise RuntimeError("HVSTER live one-minute source validity drift")
    grouped = frame.resample("5min", origin="epoch", closed="left", label="left")
    extension = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        quote_asset_volume=("quote_asset_volume", "sum"),
        number_of_trades=("number_of_trades", "sum"),
        source_rows=("open", "size"),
    ).reset_index()
    if extension.empty or not extension.source_rows.eq(5).all():
        raise RuntimeError("HVSTER live extension is not a complete one-minute-to-five-minute grid")
    return extension.drop(columns="source_rows")


def load_combined_market() -> tuple[pd.DataFrame, dict[str, Any]]:
    historical = pd.read_csv(
        MARKET,
        compression="infer",
        usecols=["date", "open", "high", "low", "close", "quote_asset_volume", "number_of_trades"],
    )
    historical["date"] = pd.to_datetime(historical.date, utc=True)
    from sqlalchemy import text

    engine = _postgres_engine()
    try:
        with engine.connect() as connection:
            raw = pd.read_sql_query(
                text(LIVE_QUERY),
                connection,
                params={"start": EXTENSION_START.to_pydatetime(), "end": END.to_pydatetime()},
            )
    finally:
        engine.dispose()
    live = _five_minute_extension(raw)
    combined = (
        pd.concat([historical, live], ignore_index=True, sort=False)
        .sort_values("date")
        .drop_duplicates("date", keep="last")
    )
    combined = combined[combined.date.lt(END)].reset_index(drop=True)
    if combined.empty or combined.date.duplicated().any():
        raise RuntimeError("HVSTER combined market identity drift")
    if not combined.date.diff().dropna().eq(pd.Timedelta(minutes=5)).all():
        raise RuntimeError("HVSTER combined market continuity drift")
    return combined, {
        "historical_path": str(MARKET),
        "historical_sha256": sha256(MARKET),
        "historical_rows": len(historical),
        "live_query_sha256": hashlib.sha256(LIVE_QUERY.encode()).hexdigest(),
        "live_one_minute_rows": len(raw),
        "live_five_minute_rows": len(live),
        "combined_rows": len(combined),
        "first": str(combined.date.iloc[0]),
        "last": str(combined.date.iloc[-1]),
        "end_exclusive": END.isoformat(),
        "mode": "hash_bound_historical_cache_plus_read_only_postgres_completed_bar_extension",
    }


def _valid_prices(window: pd.DataFrame) -> pd.Series:
    prices = window[["open", "high", "low", "close"]]
    return (
        np.isfinite(prices).all(axis=1)
        & prices.gt(0).all(axis=1)
        & window.high.ge(window[["open", "close"]].max(axis=1))
        & window.low.le(window[["open", "close"]].min(axis=1))
        & window.high.ge(window.low)
    )


def _score_anchor(window: pd.DataFrame) -> dict[str, float]:
    invalid = {
        "execution_count": float("nan"),
        "average_ticket": float("nan"),
        "block_return": float("nan"),
        "range_vol": float("nan"),
    }
    if len(window) != 144 or not bool(_valid_prices(window).all()):
        return invalid
    block = window.iloc[-72:]
    quote = pd.to_numeric(block.quote_asset_volume, errors="coerce").to_numpy(float)
    count = pd.to_numeric(block.number_of_trades, errors="coerce").to_numpy(float)
    if not np.isfinite(quote).all() or not np.isfinite(count).all() or (quote < 0).any() or (count < 0).any() or quote.sum() <= 0 or count.sum() <= 0:
        return invalid
    range_high = float(window.high.max())
    range_low = float(window.low.min())
    midpoint = 0.5 * (range_high + range_low)
    return {
        "execution_count": float(count.sum()),
        "average_ticket": float(quote.sum() / count.sum()),
        "block_return": float(np.log(float(block.close.iloc[-1]) / float(block.open.iloc[0]))),
        "range_vol": float((range_high - range_low) / midpoint),
    }


def score_snapshot(market: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    frame = market.copy()
    frame["date"] = pd.to_datetime(frame.date, utc=True)
    for column in ("open", "high", "low", "close", "quote_asset_volume", "number_of_trades"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    dates = pd.DatetimeIndex(frame.date)
    positions = np.flatnonzero(dates.minute.to_numpy() == 55).astype(np.int64)
    positions = positions[positions >= 143]
    complete = dates[positions] - dates[positions - 143] == pd.Timedelta(minutes=715)
    rows = []
    for position in positions[np.asarray(complete)]:
        rows.append(
            {
                "decision_bar_time": dates[position],
                **_score_anchor(frame.iloc[position - 143 : position + 1]),
            }
        )
    scores = pd.DataFrame(rows)
    calibration = scores[
        scores.decision_bar_time.ge(pd.Timestamp("2023-01-01T00:00:00Z"))
        & scores.decision_bar_time.lt(pd.Timestamp("2023-07-01T00:00:00Z"))
    ].replace([np.inf, -np.inf], np.nan).dropna()
    if len(calibration) < 4_000:
        raise RuntimeError("HVSTER source-only calibration floor failed")
    thresholds = {
        "execution_count_q75": float(calibration.execution_count.quantile(0.75)),
        "average_ticket_q35": float(calibration.average_ticket.quantile(0.35)),
        "absolute_block_return_q60": float(calibration.block_return.abs().quantile(0.60)),
        "range_vol_q65": float(calibration.range_vol.quantile(0.65)),
    }
    return (
        scores[scores.decision_bar_time.ge(pd.Timestamp("2023-07-01T00:00:00Z"))]
        .reset_index(drop=True),
        thresholds,
    )


def conditions(
    scores: pd.DataFrame, thresholds: dict[str, float], control: str = "primary"
) -> tuple[np.ndarray, np.ndarray]:
    if control not in ("primary", *CONTROLS):
        raise ValueError(f"unknown HVSTER control: {control}")
    execution_count = pd.to_numeric(scores.execution_count, errors="coerce").to_numpy(float)
    average_ticket = pd.to_numeric(scores.average_ticket, errors="coerce").to_numpy(float)
    volatility = pd.to_numeric(scores.range_vol, errors="coerce").to_numpy(float)
    block_return = pd.to_numeric(scores.block_return, errors="coerce").to_numpy(float)
    block_side = np.sign(block_return)
    finite = (
        np.isfinite(execution_count)
        & np.isfinite(average_ticket)
        & np.isfinite(volatility)
        & np.isfinite(block_return)
    )
    if control == "one_block_stale_participation":
        execution_count = np.r_[np.nan, execution_count[:-1]]
        average_ticket = np.r_[np.nan, average_ticket[:-1]]
    volatility_gate = volatility >= thresholds["range_vol_q65"]
    if control == "no_volatility_gate":
        volatility_gate[:] = True
    count_gate = execution_count >= thresholds["execution_count_q75"]
    ticket_gate = average_ticket <= thresholds["average_ticket_q35"]
    if control == "count_tail_only": ticket_gate[:] = True
    if control == "ticket_tail_only": count_gate[:] = True
    eligible = finite & count_gate & ticket_gate & (np.abs(block_return) >= thresholds["absolute_block_return_q60"]) & volatility_gate & (block_side != 0)
    side = (-block_side).astype(np.int8)
    previous_same_side = np.r_[False, eligible[:-1] & (side[:-1] == side[1:])]
    onset = eligible & ~previous_same_side
    if control == "direction_flip":
        side = -side
    return onset, side


def build_clock(
    scores: pd.DataFrame, thresholds: dict[str, float], control: str = "primary"
) -> pd.DataFrame:
    onset, sides = conditions(scores, thresholds, control)
    rows: list[dict[str, Any]] = []
    next_allowed: pd.Timestamp | None = None
    for index in np.flatnonzero(onset):
        decision = pd.Timestamp(scores.iloc[index].decision_bar_time)
        entry = decision + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=8)
        split = next(
            (
                name
                for name, (start, end) in SPLITS.items()
                if entry >= start and exit_time <= end
            ),
            None,
        )
        if split is None or (next_allowed is not None and entry < next_allowed):
            continue
        next_allowed = exit_time
        rows.append(
            {
                "candidate": "HVSTER-8",
                "control": control,
                "split": split,
                "decision_time": decision,
                "feature_available_time": entry,
                "entry_time": entry,
                "exit_time": exit_time,
                "side": int(sides[index]),
            }
        )
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    rows = clock[clock.split.eq(split)]
    if rows.empty:
        return {
            "events": 0,
            "longs": 0,
            "shorts": 0,
            "minority_side_share": 0.0,
            "max_month_share": 0.0,
        }
    longs = int(rows.side.eq(1).sum())
    shorts = int(rows.side.eq(-1).sum())
    months = pd.to_datetime(rows.entry_time, utc=True).dt.strftime("%Y-%m").value_counts()
    return {
        "events": len(rows),
        "longs": longs,
        "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(rows),
        "max_month_share": int(months.max()) / len(rows),
    }


def run() -> dict[str, Any]:
    if sha256(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVSTER preregistration drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    market, source = load_combined_market()
    scores, thresholds = score_snapshot(market)
    primary = build_clock(scores, thresholds)
    controls = {name: build_clock(scores, thresholds, name) for name in CONTROLS}
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(scores, SNAPSHOT)
    _write_gzip_csv(primary, CLOCK)
    for name, frame in controls.items():
        _write_gzip_csv(frame, CONTROL_DIR / f"{name}.csv.gz")
    support = {name: stats(primary, name) for name in SPLITS}
    checks = {
        check: passed
        for name, item in support.items()
        for check, passed in (
            (f"{name}_minimum_events", item["events"] >= MINIMUM[name]),
            (f"{name}_side_balance", item["minority_side_share"] >= 0.2),
            (f"{name}_month_concentration", item["max_month_share"] <= 0.45),
        )
    }
    passed = all(checks.values())
    core = {
        "protocol_version": "hvster_8_source_support_v1",
        "policy_id": "HVSTER-8",
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT),
            "sha256": PREREG_SHA,
            "manifest_hash": registration["manifest_hash"],
        },
        "source": source,
        "feature_contract": {
            "calibration_window": ["2023-01-01T00:00:00Z", "2023-07-01T00:00:00Z"],
            **thresholds,
            "ticket_definition": "block_quote_asset_volume/block_number_of_trades",
            "outcomes_opened": False,
        },
        "source_snapshot": {"path": str(SNAPSHOT), "sha256": sha256(SNAPSHOT), "rows": len(scores)},
        "completed_preentry_sources_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha256(CLOCK), "rows": len(primary)},
        "controls": {
            name: {
                "path": str(CONTROL_DIR / f"{name}.csv.gz"),
                "sha256": sha256(CONTROL_DIR / f"{name}.csv.gz"),
                "rows": len(frame),
                "promotion_authorized": False,
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
    report = {**core, "manifest_hash": prereg.canonical_hash(core)}
    RESULT.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    result = run()
    print(json.dumps({"passed": result["support_passed"], "support": result["support"]}, indent=2))
