"""Calibrate post-only maker refresh deviation bands from Binance 1m bars.

The simulation is intentionally microstructure-light because historical L2 is
not available locally.  It uses 1m OHLC as a conservative proxy: at each refresh
minute, place a post-only order near the minute open and count a fill when the
minute high/low crosses that maker price.  The selected threshold is the
smallest band within 0.5 percentage point of the maximum fill rate.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sqlalchemy import text

from preprocessing.live_db_features import sqlalchemy_engine_from_env


def calibrate(env: str, start: str, end: str) -> pd.DataFrame:
    eng = sqlalchemy_engine_from_env(env)
    with eng.connect() as conn:
        df = pd.read_sql_query(
            text(
                """
                SELECT ts, open, high, low, close
                FROM bars_binance
                WHERE symbol='BTCUSDT' AND interval='1m' AND ts >= :start AND ts <= :end
                ORDER BY ts
                """
            ),
            conn,
            params={"start": start, "end": end},
        )
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().reset_index(drop=True)

    idx = np.flatnonzero((df["ts"].dt.minute.to_numpy() % 5) == 0)
    idx = idx[idx + 10 < len(df)]
    openp = df["open"].to_numpy(float)
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    close = df["close"].to_numpy(float)
    thresholds = np.array([0, 0.0001, 0.0002, 0.0003, 0.0005, 0.00075, 0.001, 0.0015, 0.002, 0.003, 0.005, 0.0075, 0.01])
    maker_offset = 0.0001
    rows: list[dict[str, float | str | int]] = []
    for kind, horizon in [("open", 5), ("close", 10)]:
        refs = close[idx]
        future_open = np.stack([openp[idx + j] for j in range(1, horizon + 1)], axis=1)
        future_high = np.stack([high[idx + j] for j in range(1, horizon + 1)], axis=1)
        future_low = np.stack([low[idx + j] for j in range(1, horizon + 1)], axis=1)
        drift = np.abs(future_open / refs[:, None] - 1.0)
        for side in ["BUY", "SELL"]:
            fillable = future_low <= future_open * (1 - maker_offset) if side == "BUY" else future_high >= future_open * (1 + maker_offset)
            for threshold in thresholds:
                eligible = drift <= threshold
                fill = (eligible & fillable).any(axis=1)
                rows.append(
                    {
                        "kind": kind,
                        "side": side,
                        "threshold": float(threshold),
                        "fill_rate": float(fill.mean()),
                        "attempts_per_event": float(eligible.sum(axis=1).mean()),
                        "fill_count": int(fill.sum()),
                        "events": int(len(idx)),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env", default=".env")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="2026-07-07 16:00")
    args = p.parse_args()
    res = calibrate(args.env, args.start, args.end)
    print(res.to_string(index=False))
    for kind in ["open", "close"]:
        sub = res[res.kind == kind].groupby("threshold").agg(fill_rate=("fill_rate", "mean"), attempts=("attempts_per_event", "mean")).reset_index()
        max_fill = float(sub.fill_rate.max())
        selected = sub[sub.fill_rate >= max_fill - 0.005].iloc[0]
        print(f"SELECT {kind} threshold={float(selected.threshold):.6f} fill_rate={float(selected.fill_rate):.6f} maxfill={max_fill:.6f} attempts={float(selected.attempts):.3f}")


if __name__ == "__main__":
    main()
