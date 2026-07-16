"""Build PFCR-1's outcome-blind 2023-2024 event clock."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.export_leave_one_out_residual_exhaustion_sources import (
    deterministic_csv_gz,
    sha256_file,
)
from training.preregister_post_funding_cross_sectional_crowding_release import (
    canonical_hash,
    protocol,
)


START = pd.Timestamp("2023-01-01 00:00:00")
END = pd.Timestamp("2025-01-01 00:00:00")
SYMBOLS = ("ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT")
SOURCE_DIR = Path("data/binance_um_lore_2023_2024")
PREREGISTRATION = Path(
    "results/post_funding_cross_sectional_crowding_release_preregistration_2026-07-17.json"
)
EXPECTED_PREREGISTRATION_SHA256 = "0863d5464a5f74940e3aa8e76b90bc5306efda914fa3fad4fc2c36eccaa3c6a0"
EXPECTED_PROTOCOL_HASH = "bc68b2788d1a67f28fad7744ff480af58da0a111a2fcd48cc42b4288d4e57528"
DEFAULT_CLOCK = Path("data/post_funding_cross_sectional_crowding_release_clock_2023_2024.csv.gz")
DEFAULT_MANIFEST = Path(
    "results/post_funding_cross_sectional_crowding_release_support_2026-07-17.json"
)
DEFAULT_DOCS = Path("docs/post-funding-cross-sectional-crowding-release-support-2026-07-17.md")
POLICY_ID = "PFCR01"
SPREAD_LOOKBACK = 180
SPREAD_MINIMUM = 90
SPREAD_QUANTILE = 0.90
BETA_LOOKBACK_HOURS = 720
BETA_MINIMUM_HOURS = 336
HOLD = pd.Timedelta(hours=4)


CLOCK_COLUMNS = (
    "policy_id",
    "settlement_time",
    "feature_available_time",
    "entry_time",
    "exit_time",
    "long_symbol",
    "short_symbol",
    "long_weight",
    "short_weight_abs",
    "long_beta",
    "short_beta",
    "current_funding_spread",
    "prior_spread_q90",
    "long_current_funding_rate",
    "short_current_funding_rate",
)


def _verify_preregistration() -> dict[str, Any]:
    if sha256_file(PREREGISTRATION) != EXPECTED_PREREGISTRATION_SHA256:
        raise RuntimeError("PFCR preregistration file changed")
    payload = json.loads(PREREGISTRATION.read_text())
    if payload.get("protocol_hash") != EXPECTED_PROTOCOL_HASH:
        raise RuntimeError("PFCR preregistration identity changed")
    if canonical_hash(payload["protocol"]) != EXPECTED_PROTOCOL_HASH:
        raise RuntimeError("PFCR preregistration body changed")
    if canonical_hash(protocol()) != EXPECTED_PROTOCOL_HASH:
        raise RuntimeError("PFCR implementation protocol drifted")
    if payload["protocol"]["evidence_boundary"]["exact_pfcr_pair_clock_or_returns_opened"]:
        raise RuntimeError("PFCR preregistration already opened the family")
    return payload


def _market_path(symbol: str) -> Path:
    return SOURCE_DIR / f"{symbol}_5m_2023_2024.csv.gz"


def _funding_path(symbol: str) -> Path:
    return SOURCE_DIR / f"{symbol}_funding_2023_2024.csv.gz"


def load_feature_sources() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    close: dict[str, pd.Series] = {}
    funding: dict[str, pd.Series] = {}
    records: dict[str, Any] = {}
    expected_5m = pd.date_range(START, END - pd.Timedelta(minutes=5), freq="5min")
    expected_hourly = pd.date_range(START + pd.Timedelta(hours=1), END, freq="1h")
    for symbol in SYMBOLS:
        market_path = _market_path(symbol)
        funding_path = _funding_path(symbol)
        market = pd.read_csv(
            market_path,
            usecols=["date", "close", "tic"],
            parse_dates=["date"],
        )
        if not pd.DatetimeIndex(market["date"]).equals(expected_5m):
            raise RuntimeError(f"{symbol} PFCR market grid failure")
        if not market["tic"].astype(str).eq(symbol).all():
            raise RuntimeError(f"{symbol} PFCR market identity failure")
        if not np.isfinite(market["close"].to_numpy(dtype=float)).all() or (
            market["close"] <= 0.0
        ).any():
            raise RuntimeError(f"{symbol} PFCR invalid market close")
        hourly = market.set_index("date")["close"].resample(
            "1h", closed="left", label="right"
        ).last()
        if not hourly.index.equals(expected_hourly) or hourly.isna().any():
            raise RuntimeError(f"{symbol} PFCR hourly close failure")
        close[symbol] = hourly

        frame = pd.read_csv(
            funding_path,
            usecols=["symbol", "funding_time", "funding_rate"],
        )
        if not frame["symbol"].astype(str).eq(symbol).all():
            raise RuntimeError(f"{symbol} PFCR funding identity failure")
        frame["settlement_time"] = pd.to_datetime(
            pd.to_numeric(frame["funding_time"], errors="raise"), unit="ms"
        )
        frame["funding_rate"] = pd.to_numeric(frame["funding_rate"], errors="raise")
        if frame["settlement_time"].duplicated().any() or not frame[
            "settlement_time"
        ].is_monotonic_increasing:
            raise RuntimeError(f"{symbol} PFCR funding order failure")
        funding[symbol] = pd.Series(
            frame["funding_rate"].to_numpy(dtype=float),
            index=pd.DatetimeIndex(frame["settlement_time"]),
            name=symbol,
        )
        records[symbol] = {
            "market_path": str(market_path),
            "market_sha256": sha256_file(market_path),
            "funding_path": str(funding_path),
            "funding_sha256": sha256_file(funding_path),
            "market_rows": int(len(market)),
            "funding_rows": int(len(frame)),
        }
    close_frame = pd.DataFrame(close)
    funding_frame = pd.concat(funding.values(), axis=1, join="inner").sort_index()
    if list(funding_frame.columns) != list(SYMBOLS):
        raise RuntimeError("PFCR funding column order drift")
    if funding_frame.isna().any().any() or len(funding_frame) != len(funding[SYMBOLS[0]]):
        raise RuntimeError("PFCR funding settlements are not common across six symbols")
    if any(not funding_frame.index.equals(series.index) for series in funding.values()):
        raise RuntimeError("PFCR exact funding timestamps differ by symbol")
    return close_frame, funding_frame, records


def causal_betas(close: pd.DataFrame) -> pd.DataFrame:
    returns = np.log(close).diff()
    beta = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    for symbol in SYMBOLS:
        factor = returns.drop(columns=symbol).median(axis=1)
        covariance = returns[symbol].rolling(
            BETA_LOOKBACK_HOURS, min_periods=BETA_MINIMUM_HOURS
        ).cov(factor)
        variance = factor.rolling(
            BETA_LOOKBACK_HOURS, min_periods=BETA_MINIMUM_HOURS
        ).var()
        beta[symbol] = (covariance / variance.replace(0.0, np.nan)).shift(1).clip(
            0.25, 2.5
        )
    return beta


def build_clock(funding: pd.DataFrame, beta: pd.DataFrame) -> pd.DataFrame:
    spread = funding.max(axis=1) - funding.min(axis=1)
    threshold = spread.rolling(SPREAD_LOOKBACK, min_periods=SPREAD_MINIMUM).quantile(
        SPREAD_QUANTILE
    ).shift(1)
    rows: list[dict[str, Any]] = []
    previous_exit: pd.Timestamp | None = None
    for settlement, rates in funding.iterrows():
        event = pd.Timestamp(settlement)
        prior_q90 = float(threshold.loc[event])
        current_spread = float(spread.loc[event])
        if not np.isfinite(prior_q90) or current_spread <= max(prior_q90, 0.0):
            continue
        if event not in beta.index:
            continue
        current_beta = beta.loc[event, list(SYMBOLS)]
        if not np.isfinite(current_beta.to_numpy(dtype=float)).all():
            continue
        minimum_rate = float(rates.min())
        maximum_rate = float(rates.max())
        long_symbol = sorted(rates.index[rates.eq(minimum_rate)].astype(str))[0]
        short_symbol = sorted(rates.index[rates.eq(maximum_rate)].astype(str))[0]
        if long_symbol == short_symbol:
            continue
        long_beta = float(current_beta[long_symbol])
        short_beta = float(current_beta[short_symbol])
        denominator = long_beta + short_beta
        long_weight = short_beta / denominator
        short_weight = long_beta / denominator
        available = event + pd.Timedelta(minutes=5)
        entry = event + pd.Timedelta(minutes=10)
        exit_time = entry + HOLD
        if entry < START or exit_time >= END:
            continue
        if previous_exit is not None and entry < previous_exit:
            continue
        rows.append(
            {
                "policy_id": POLICY_ID,
                "settlement_time": event,
                "feature_available_time": available,
                "entry_time": entry,
                "exit_time": exit_time,
                "long_symbol": long_symbol,
                "short_symbol": short_symbol,
                "long_weight": long_weight,
                "short_weight_abs": short_weight,
                "long_beta": long_beta,
                "short_beta": short_beta,
                "current_funding_spread": current_spread,
                "prior_spread_q90": prior_q90,
                "long_current_funding_rate": minimum_rate,
                "short_current_funding_rate": maximum_rate,
            }
        )
        previous_exit = exit_time
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def assert_clock_contract(clock: pd.DataFrame) -> None:
    if clock.empty:
        return
    for column in ("settlement_time", "feature_available_time", "entry_time", "exit_time"):
        clock[column] = pd.to_datetime(clock[column], errors="raise")
    if not clock["policy_id"].eq(POLICY_ID).all():
        raise RuntimeError("PFCR policy identity changed")
    if not (
        clock["feature_available_time"] == clock["settlement_time"] + pd.Timedelta(minutes=5)
    ).all():
        raise RuntimeError("PFCR feature latency changed")
    if not (
        clock["entry_time"] == clock["settlement_time"] + pd.Timedelta(minutes=10)
    ).all():
        raise RuntimeError("PFCR entry latency changed")
    if not (clock["exit_time"] == clock["entry_time"] + HOLD).all():
        raise RuntimeError("PFCR hold changed")
    if not (clock["feature_available_time"] < clock["entry_time"]).all():
        raise RuntimeError("PFCR feature crossed entry")
    if not (clock["current_funding_spread"] > clock["prior_spread_q90"]).all():
        raise RuntimeError("PFCR spread gate changed")
    if not np.allclose(clock["long_weight"] + clock["short_weight_abs"], 1.0):
        raise RuntimeError("PFCR gross-one sizing changed")
    exposure = clock["long_weight"] * clock["long_beta"] - clock[
        "short_weight_abs"
    ] * clock["short_beta"]
    if not np.allclose(exposure, 0.0, atol=1e-12):
        raise RuntimeError("PFCR beta neutrality changed")
    if (
        clock["entry_time"].iloc[1:].reset_index(drop=True)
        < clock["exit_time"].iloc[:-1].reset_index(drop=True)
    ).any():
        raise RuntimeError("PFCR clock overlaps")
    forbidden = {"pnl", "profit", "target", "label", "future", "cagr", "mdd"}
    for column in clock.columns:
        if set(column.lower().split("_")) & forbidden:
            raise RuntimeError(f"PFCR outcome-like clock column forbidden: {column}")


def support_stats(clock: pd.DataFrame) -> dict[str, Any]:
    n = len(clock)
    if n:
        pair_counts = Counter(clock["short_symbol"] + ">" + clock["long_symbol"])
        year_counts = clock.groupby(clock["settlement_time"].dt.year).size().to_dict()
        month_counts = clock.groupby(clock["settlement_time"].dt.to_period("M")).size()
        maximum_pair_share = max(pair_counts.values()) / n
        maximum_month_share = int(month_counts.max()) / n
        long_symbols = sorted(clock["long_symbol"].unique())
        short_symbols = sorted(clock["short_symbol"].unique())
    else:
        pair_counts, year_counts = Counter(), {}
        maximum_pair_share = maximum_month_share = 1.0
        long_symbols, short_symbols = [], []
    gates = {
        "events_2023_2024_at_least_60": n >= 60,
        "events_each_year_at_least_25": all(year_counts.get(year, 0) >= 25 for year in (2023, 2024)),
        "unique_ordered_pairs_at_least_8": len(pair_counts) >= 8,
        "maximum_ordered_pair_share_at_most_0_25": maximum_pair_share <= 0.25,
        "long_symbols_at_least_4": len(long_symbols) >= 4,
        "short_symbols_at_least_4": len(short_symbols) >= 4,
        "maximum_month_share_at_most_0_20": maximum_month_share <= 0.20,
    }
    return {
        "events": n,
        "year_counts": {str(key): int(value) for key, value in year_counts.items()},
        "unique_ordered_pairs": len(pair_counts),
        "ordered_pair_counts": dict(sorted(pair_counts.items())),
        "maximum_ordered_pair_share": maximum_pair_share,
        "maximum_month_share": maximum_month_share,
        "long_symbols": long_symbols,
        "short_symbols": short_symbols,
        "gates": gates,
        "passes_support": all(gates.values()),
    }


def _markdown(result: dict[str, Any]) -> str:
    stats = result["support"]
    return f"""# PFCR-1 outcome-blind support — 2026-07-17

- Post-entry returns/PnL calculated: **no**
- Common settlements: `{result['common_settlements']}`
- Eligible events: `{stats['events']}`; years `{stats['year_counts']}`
- Unique ordered pairs: `{stats['unique_ordered_pairs']}`
- Maximum pair share: `{stats['maximum_ordered_pair_share']:.2%}`
- Maximum month share: `{stats['maximum_month_share']:.2%}`
- Long symbols: `{stats['long_symbols']}`
- Short symbols: `{stats['short_symbols']}`
- Support gate: **{'PASS' if stats['passes_support'] else 'REJECT'}**
- Clock SHA-256: `{result['clock_sha256']}`

The clock uses current settled funding only after the exact common timestamp,
a strictly prior rolling 180-event q90 spread threshold, causal shifted betas,
feature availability at +5 minutes, entry at +10 minutes, and a fixed four-
hour hold. No 2025 or 2026 source was read.
"""


def run(
    clock_path: str | Path = DEFAULT_CLOCK,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    docs_path: str | Path = DEFAULT_DOCS,
) -> dict[str, Any]:
    preregistration = _verify_preregistration()
    close, funding, records = load_feature_sources()
    beta = causal_betas(close)
    clock = build_clock(funding, beta)
    assert_clock_contract(clock)
    stats = support_stats(clock)
    clock_output = Path(clock_path)
    deterministic_csv_gz(clock, clock_output)
    reread = pd.read_csv(clock_output)
    assert_clock_contract(reread)
    result: dict[str, Any] = {
        "protocol_version": "pfcr_v1_support_2026-07-17",
        "preregistration_protocol_hash": preregistration["protocol_hash"],
        "post_entry_returns_calculated": False,
        "2023_fit_opened": False,
        "2024_test_opened": False,
        "2025_eval_opened": False,
        "2026_holdout_opened": False,
        "common_settlements": int(len(funding)),
        "clock_path": str(clock_output),
        "clock_sha256": sha256_file(clock_output),
        "source_records": records,
        "support": stats,
    }
    body = dict(result)
    result["manifest_hash"] = canonical_hash(body)
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    manifest_output = Path(manifest_path)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    docs_output = Path(docs_path)
    docs_output.parent.mkdir(parents=True, exist_ok=True)
    docs_output.write_text(_markdown(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clock", default=str(DEFAULT_CLOCK))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--docs", default=str(DEFAULT_DOCS))
    args = parser.parse_args()
    print(json.dumps(run(args.clock, args.manifest, args.docs), indent=2))


if __name__ == "__main__":
    main()
