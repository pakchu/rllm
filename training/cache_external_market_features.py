"""Cache market bars with wave_trading external features attached."""
from __future__ import annotations

import argparse, json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from preprocessing.external_features import attach_wave_trading_external_features


def cache_external_market_features(input_csv: str, output: str, wave_trading_root: str, external_tolerance: str = "30min", start: str = "", end: str = "") -> dict:
    df = pd.read_csv(input_csv, parse_dates=["date"], compression="infer")
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="raise").dt.tz_convert(None)
    df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    if start or end:
        # Keep warmup before start for rolling features while reducing file size.
        if start:
            start_ts = pd.Timestamp(start) - pd.Timedelta(days=3)
            df = df[df["date"] >= start_ts]
        if end:
            df = df[df["date"] <= pd.Timestamp(end)]
        df = df.reset_index(drop=True)
    out = attach_wave_trading_external_features(df, wave_trading_root=wave_trading_root, tolerance=external_tolerance)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    if str(output).endswith(".parquet"):
        out.to_parquet(output, index=False)
    else:
        out.to_csv(output, index=False, compression="gzip" if str(output).endswith(".gz") else None)
    nonzero_after = 0
    if "kimchi_premium_change" in out.columns:
        nonzero_after = int(((out["date"] > pd.Timestamp("2025-12-15")) & (out["kimchi_premium_change"].abs() > 1e-12)).sum())
    return {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "input_csv": input_csv,
        "output": output,
        "rows": int(len(out)),
        "date_min": str(out["date"].min()) if len(out) else None,
        "date_max": str(out["date"].max()) if len(out) else None,
        "columns": list(out.columns),
        "kimchi_nonzero_after_2025_12_15": nonzero_after,
    }


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--input-csv', required=True); p.add_argument('--output', required=True); p.add_argument('--wave-trading-root', required=True)
    p.add_argument('--external-tolerance', default='30min'); p.add_argument('--start', default=''); p.add_argument('--end', default='')
    return p.parse_args()


def main():
    print(json.dumps(cache_external_market_features(**vars(parse_args())), indent=2, ensure_ascii=False))

if __name__=='__main__': main()
