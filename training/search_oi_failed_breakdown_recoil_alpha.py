"""Stage-gated search for the preregistered OI failed-breakdown recoil alpha.

OFBR-1 is a long-only lifecycle event:

1. price sells off into the lower part of its trailing range while delayed OI
   expands relative to price and taker flow is seller-dominant;
2. after a fixed wait, price reclaims the anchor, fast taker flow improves,
   and delayed OI remains in place; and
3. enter at the next five-minute open for a fixed hold.

The parameter grid and stage boundaries are frozen in
``results/oi_failed_breakdown_recoil_preregistration_2026-07-28.json``.
``pre2024`` never loads a row at or after 2024. ``test2024`` consumes only the
frozen pre-2024 shortlist and cannot open 2025+. Future evaluation is handled
by a later one-shot portfolio verifier after a unique 2024 top-1 is frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PREREGISTRATION = Path(
    "results/oi_failed_breakdown_recoil_preregistration_2026-07-28.json"
)
PRE2024_OUTPUT = Path(
    "results/oi_failed_breakdown_recoil_pre2024_selection_2026-07-28.json"
)
PRE2024_DOC = Path(
    "docs/oi-failed-breakdown-recoil-pre2024-selection-2026-07-28.md"
)
TEST2024_OUTPUT = Path(
    "results/oi_failed_breakdown_recoil_test2024_selection_2026-07-28.json"
)
TEST2024_DOC = Path(
    "docs/oi-failed-breakdown-recoil-test2024-selection-2026-07-28.md"
)

WINDOWS = {
    "calibration": ("2020-09-01", "2023-01-01"),
    "calibration_2020q4": ("2020-09-01", "2021-01-01"),
    "calibration_2021": ("2021-01-01", "2022-01-01"),
    "calibration_2022": ("2022-01-01", "2023-01-01"),
    "train_2023": ("2023-01-01", "2024-01-01"),
    "train_2023_h1": ("2023-01-01", "2023-07-01"),
    "train_2023_h2": ("2023-07-01", "2024-01-01"),
    "test_2024": ("2024-01-01", "2025-01-01"),
    "test_2024_h1": ("2024-01-01", "2024-07-01"),
    "test_2024_h2": ("2024-07-01", "2025-01-01"),
}


@dataclass(frozen=True)
class Config:
    market_csv: str = (
        "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz"
    )
    oi_csv: str = "/tmp/btcusdt_open_interest_5m_2020_2026.csv"
    preregistration: str = str(PREREGISTRATION)
    pre2024_output: str = str(PRE2024_OUTPUT)
    pre2024_docs: str = str(PRE2024_DOC)
    test2024_output: str = str(TEST2024_OUTPUT)
    test2024_docs: str = str(TEST2024_DOC)
    leverage: float = 0.5
    cost_rate: float = 0.0006
    stress_cost_rate: float = 0.001


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_preregistration(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("name") != "oi_failed_breakdown_recoil":
        raise RuntimeError("unexpected preregistration name")
    if payload.get("status") != "preregistered_before_outcome_scan":
        raise RuntimeError("preregistration is not outcome-blind")
    if payload["frozen_grid"].get("candidate_count") != 96:
        raise RuntimeError("frozen candidate count drifted")
    if payload["causality_contract"].get("future_can_rerank") is not False:
        raise RuntimeError("future reranking is not disabled")
    return payload


def _read_market_prefix(path: str | Path, cutoff: str) -> pd.DataFrame:
    """Read only rows before cutoff into the returned frame.

    Pandas may decompress a chunk that crosses the boundary. The returned frame,
    feature graph, and every outcome calculation are hard-filtered before the
    cutoff, matching the repository's disclosed compressed-source contract.
    """
    chunks: list[pd.DataFrame] = []
    columns = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "quote_asset_volume",
        "taker_buy_quote",
    ]
    limit = pd.Timestamp(cutoff)
    for chunk in pd.read_csv(
        path,
        usecols=columns,
        parse_dates=["date"],
        compression="infer",
        chunksize=100_000,
    ):
        dates = pd.to_datetime(chunk["date"], utc=True, errors="raise").dt.tz_convert(None)
        chunk = chunk.assign(date=dates)
        keep = chunk["date"] < limit
        if keep.any():
            chunks.append(chunk.loc[keep])
        if (~keep).any() or (len(chunk) and chunk["date"].iloc[-1] >= limit):
            break
    if not chunks:
        raise RuntimeError("market prefix is empty")
    market = pd.concat(chunks, ignore_index=True)
    return (
        market.sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )


def _read_oi_prefix(path: str | Path, cutoff: str) -> pd.DataFrame:
    limit = pd.Timestamp(cutoff)
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=["date", "open_interest"], chunksize=100_000):
        dates = pd.to_datetime(chunk["date"], utc=True, errors="raise").dt.tz_convert(None)
        chunk = chunk.assign(date=dates)
        keep = chunk["date"] < limit
        if keep.any():
            chunks.append(chunk.loc[keep])
        if (~keep).any() or (len(chunk) and chunk["date"].iloc[-1] >= limit):
            break
    if not chunks:
        raise RuntimeError("OI prefix is empty")
    oi = pd.concat(chunks, ignore_index=True)
    return oi.sort_values("date").drop_duplicates("date", keep="last")


def load_sources(cfg: Config, *, cutoff: str) -> pd.DataFrame:
    market = _read_market_prefix(cfg.market_csv, cutoff)
    oi = _read_oi_prefix(cfg.oi_csv, cutoff)
    merged = market.merge(oi, on="date", how="left", validate="one_to_one")
    if len(merged) and pd.Timestamp(merged["date"].max()) >= pd.Timestamp(cutoff):
        raise RuntimeError("source cutoff was not enforced")
    return merged.reset_index(drop=True)


def build_features(market: pd.DataFrame) -> pd.DataFrame:
    """Build the frozen completed-bar features with one-bar-delayed OI."""
    close = pd.to_numeric(market["close"], errors="coerce")
    high = pd.to_numeric(market["high"], errors="coerce")
    low = pd.to_numeric(market["low"], errors="coerce")
    quote = pd.to_numeric(market["quote_asset_volume"], errors="coerce")
    taker_buy = pd.to_numeric(market["taker_buy_quote"], errors="coerce")

    # Exact one-bar delay and no fill: an absent live OI observation fails closed.
    oi = pd.to_numeric(market["open_interest"], errors="coerce").shift(1)
    log_close = np.log(close.where(close > 0.0))
    log_oi = np.log(oi.where(oi > 0.0))
    price_ret_4h = log_close - log_close.shift(48)
    oi_ret_4h = log_oi - log_oi.shift(48)
    imbalance = (2.0 * taker_buy / quote.replace(0.0, np.nan) - 1.0).clip(-1.0, 1.0)

    trailing_high = high.rolling(48, min_periods=48).max()
    trailing_low = low.rolling(48, min_periods=48).min()
    span = (trailing_high - trailing_low).replace(0.0, np.nan)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return pd.DataFrame(
        {
            "price_ret_4h": price_ret_4h,
            "oi_minus_price_4h": oi_ret_4h - price_ret_4h,
            "flow_1h": imbalance.rolling(12, min_periods=12).mean(),
            "flow_15m": imbalance.rolling(3, min_periods=3).mean(),
            "range_position_4h": (close - trailing_low) / span,
            "atr_1h": true_range.rolling(12, min_periods=12).mean(),
            "observed_oi": oi,
            "oi_available": oi.notna(),
        },
        index=market.index,
    ).replace([np.inf, -np.inf], np.nan)


def _quantile(values: pd.Series, mask: np.ndarray, q: float) -> float:
    sample = values.to_numpy(float)[mask]
    sample = sample[np.isfinite(sample)]
    if len(sample) < 10_000:
        raise RuntimeError(f"insufficient calibration support for q{q}: {len(sample)}")
    return float(np.quantile(sample, q))


def fit_thresholds(
    features: pd.DataFrame,
    dates: pd.Series,
    preregistration: dict[str, Any],
) -> dict[str, Any]:
    start, end = preregistration["causality_contract"]["threshold_fit_window"]
    mask = np.asarray(
        (dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end)), dtype=bool
    )
    grid = preregistration["frozen_grid"]
    return {
        "price_return": {
            str(q): _quantile(features["price_ret_4h"], mask, float(q))
            for q in grid["price_return_quantile"]
        },
        "oi_minus_price": {
            str(q): _quantile(features["oi_minus_price_4h"], mask, float(q))
            for q in grid["oi_minus_price_quantile"]
        },
        "taker_sell": {
            str(q): _quantile(features["flow_1h"], mask, float(q))
            for q in grid["taker_sell_quantile"]
        },
        "range_position": {
            str(q): _quantile(features["range_position_4h"], mask, float(q))
            for q in grid["range_position_quantile"]
        },
    }


def candidate_grid(preregistration: dict[str, Any]) -> list[dict[str, Any]]:
    grid = preregistration["frozen_grid"]
    keys = (
        "price_return_quantile",
        "oi_minus_price_quantile",
        "taker_sell_quantile",
        "range_position_quantile",
        "confirmation_wait_bars",
        "minimum_oi_retention",
        "minimum_post_anchor_range_position",
        "hold_bars",
    )
    rows: list[dict[str, Any]] = []
    for values in itertools.product(*(grid[key] for key in keys)):
        row = dict(zip(keys, values))
        row["name"] = (
            "ofbr"
            f"_d{int(round(float(row['oi_minus_price_quantile']) * 100))}"
            f"_f{int(round(float(row['taker_sell_quantile']) * 100))}"
            f"_w{int(row['confirmation_wait_bars'])}"
            f"_oi{int(round(float(row['minimum_oi_retention']) * 10000))}"
            f"_r{int(round(float(row['minimum_post_anchor_range_position']) * 100))}"
            f"_h{int(row['hold_bars'])}"
        )
        rows.append(row)
    if len(rows) != int(grid["candidate_count"]):
        raise RuntimeError("candidate grid size drifted")
    return rows


def signal_positions(
    market: pd.DataFrame,
    features: pd.DataFrame,
    thresholds: dict[str, Any],
    candidate: dict[str, Any],
    preregistration: dict[str, Any],
) -> np.ndarray:
    """Return confirmation-bar positions for one frozen candidate."""
    price_q = str(candidate["price_return_quantile"])
    div_q = str(candidate["oi_minus_price_quantile"])
    flow_q = str(candidate["taker_sell_quantile"])
    range_q = str(candidate["range_position_quantile"])
    stride = int(preregistration["frozen_grid"]["anchor_stride_bars"])
    wait = int(candidate["confirmation_wait_bars"])
    atr_multiple = float(preregistration["frozen_grid"]["post_anchor_atr_multiple"])

    price = features["price_ret_4h"].to_numpy(float)
    divergence = features["oi_minus_price_4h"].to_numpy(float)
    slow_flow = features["flow_1h"].to_numpy(float)
    fast_flow = features["flow_15m"].to_numpy(float)
    range_position = features["range_position_4h"].to_numpy(float)
    observed_oi = features["observed_oi"].to_numpy(float)
    atr = features["atr_1h"].to_numpy(float)
    close = pd.to_numeric(market["close"], errors="coerce").to_numpy(float)
    high = pd.to_numeric(market["high"], errors="coerce").to_numpy(float)
    low = pd.to_numeric(market["low"], errors="coerce").to_numpy(float)

    anchors = np.arange(48, max(48, len(market) - wait), stride, dtype=np.int64)
    active = (
        np.isfinite(price[anchors])
        & np.isfinite(divergence[anchors])
        & np.isfinite(slow_flow[anchors])
        & np.isfinite(range_position[anchors])
        & np.isfinite(observed_oi[anchors])
        & np.isfinite(atr[anchors])
        & (price[anchors] <= float(thresholds["price_return"][price_q]))
        & (divergence[anchors] >= float(thresholds["oi_minus_price"][div_q]))
        & (slow_flow[anchors] <= float(thresholds["taker_sell"][flow_q]))
        & (range_position[anchors] <= float(thresholds["range_position"][range_q]))
    )
    confirmations: list[int] = []
    retention_floor = float(candidate["minimum_oi_retention"])
    recovery_floor = float(candidate["minimum_post_anchor_range_position"])
    for anchor in anchors[active]:
        anchor = int(anchor)
        confirmation = anchor + wait
        if confirmation >= len(market):
            continue
        values = (
            close[anchor],
            close[confirmation],
            fast_flow[confirmation],
            observed_oi[anchor],
            observed_oi[confirmation],
            atr[anchor],
        )
        if not all(np.isfinite(value) for value in values):
            continue
        post_low = float(np.nanmin(low[anchor + 1 : confirmation + 1]))
        post_high = float(np.nanmax(high[anchor + 1 : confirmation + 1]))
        if not (np.isfinite(post_low) and np.isfinite(post_high) and post_high > post_low):
            continue
        post_position = (float(close[confirmation]) - post_low) / (post_high - post_low)
        oi_retention = float(observed_oi[confirmation] / observed_oi[anchor] - 1.0)
        if (
            float(close[confirmation]) > float(close[anchor])
            and float(fast_flow[confirmation]) > float(slow_flow[anchor])
            and oi_retention >= retention_floor
            and post_low >= float(close[anchor]) - atr_multiple * float(atr[anchor])
            and post_position >= recovery_floor
        ):
            confirmations.append(confirmation)
    return np.asarray(sorted(set(confirmations)), dtype=np.int64)


def _years(start: str, end: str) -> float:
    return (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / (
        365.25 * 86_400.0
    )


def simulate(
    market: pd.DataFrame,
    signals: Iterable[int],
    *,
    start: str,
    end: str,
    hold_bars: int,
    leverage: float,
    cost_rate: float,
) -> dict[str, Any]:
    """Replay one long sleeve on the same upper-before-lower bar clock as G9."""
    dates = pd.DatetimeIndex(pd.to_datetime(market["date"]))
    start_pos = int(dates.searchsorted(pd.Timestamp(start), side="left"))
    end_pos = int(dates.searchsorted(pd.Timestamp(end), side="left"))
    length = max(0, end_pos - start_pos)
    returns = np.zeros(length, dtype=np.float64)
    market_low = np.zeros(length, dtype=np.float64)
    market_high = np.zeros(length, dtype=np.float64)
    opens = pd.to_numeric(market["open"], errors="coerce").to_numpy(float)
    highs = pd.to_numeric(market["high"], errors="coerce").to_numpy(float)
    lows = pd.to_numeric(market["low"], errors="coerce").to_numpy(float)
    cost = float(cost_rate) * float(leverage)
    trades = wins = 0
    trade_returns: list[float] = []
    accepted: list[int] = []
    next_allowed = start_pos
    for raw_signal in sorted(set(int(value) for value in signals)):
        signal = int(raw_signal)
        entry = signal + 1
        exit_pos = entry + int(hold_bars)
        if (
            signal < start_pos
            or signal >= end_pos
            or signal < next_allowed
            or entry < start_pos
            or exit_pos >= end_pos
            or exit_pos >= len(market)
        ):
            continue
        local_entry = entry - start_pos
        returns[local_entry] -= cost
        trade_factor = 1.0 - cost
        for position in range(entry, exit_pos):
            open_price = float(opens[position])
            if not np.isfinite(open_price) or open_price <= 0.0:
                continue
            local = position - start_pos
            bar_return = float(leverage) * (
                float(opens[position + 1]) / open_price - 1.0
            )
            adverse = float(leverage) * (float(lows[position]) / open_price - 1.0)
            favorable = float(leverage) * (float(highs[position]) / open_price - 1.0)
            returns[local] += bar_return
            market_low[local] += min(0.0, adverse)
            market_high[local] += max(0.0, favorable)
            trade_factor *= max(0.0, 1.0 + bar_return)
        local_exit = exit_pos - start_pos
        returns[local_exit] -= cost
        trade_factor *= max(0.0, 1.0 - cost)
        realized = trade_factor - 1.0
        trades += 1
        wins += int(realized > 0.0)
        trade_returns.append(float(realized))
        accepted.append(signal)
        next_allowed = exit_pos

    if length:
        equity_after = np.cumprod(np.maximum(0.0, 1.0 + returns))
        equity_before = np.r_[1.0, equity_after[:-1]]
        low_value = equity_before * np.maximum(0.0, 1.0 + market_low)
        high_value = equity_before * np.maximum(0.0, 1.0 + market_high)
        upper = np.maximum.reduce(
            [equity_before, equity_after, low_value, high_value]
        )
        peak = np.maximum.accumulate(upper)
        lower = np.minimum.reduce(
            [equity_before, equity_after, low_value, high_value]
        )
        mdd = float(np.max(1.0 - lower / np.maximum(peak, 1e-12))) * 100.0
        final = float(equity_after[-1])
    else:
        mdd = 0.0
        final = 1.0
    absolute_return = (final - 1.0) * 100.0
    years = _years(start, end)
    cagr = (final ** (1.0 / years) - 1.0) * 100.0 if final > 0.0 else -100.0
    sample = np.asarray(trade_returns, dtype=float)
    if len(sample) > 1 and float(np.std(sample, ddof=1)) > 0.0:
        t_stat = float(np.mean(sample) / (np.std(sample, ddof=1) / math.sqrt(len(sample))))
    else:
        t_stat = 0.0
    return {
        "absolute_return_pct": absolute_return,
        "cagr_pct": cagr,
        "strict_mdd_pct": mdd,
        "cagr_to_strict_mdd": cagr / mdd if mdd > 1e-12 else 0.0,
        "trades": trades,
        "wins": wins,
        "win_rate": wins / trades if trades else 0.0,
        "mean_trade_bps": float(sample.mean() * 10_000.0) if len(sample) else 0.0,
        "trade_t_stat": t_stat,
        "accepted_signal_positions": accepted,
        "signal_hash": hashlib.sha256(
            np.asarray(accepted, dtype="<i8").tobytes()
        ).hexdigest(),
    }


def _public_metric(metric: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metric.items() if key != "accepted_signal_positions"}


def _pre2024_passes(stats: dict[str, dict[str, Any]]) -> bool:
    calibration = stats["calibration"]
    train = stats["train_2023"]
    return bool(
        calibration["absolute_return_pct"] > 0.0
        and train["absolute_return_pct"] > 0.0
        and train["trades"] >= 12
        and calibration["strict_mdd_pct"] <= 20.0
        and train["strict_mdd_pct"] <= 20.0
        and min(
            calibration["cagr_to_strict_mdd"],
            train["cagr_to_strict_mdd"],
        )
        >= 1.0
    )


def _pre2024_key(row: dict[str, Any]) -> tuple[Any, ...]:
    stats = row["stats"]
    calibration = stats["calibration"]
    train = stats["train_2023"]
    parameters = row["candidate"]
    return (
        min(
            calibration["cagr_to_strict_mdd"],
            train["cagr_to_strict_mdd"],
        ),
        train["cagr_to_strict_mdd"],
        -max(calibration["strict_mdd_pct"], train["strict_mdd_pct"]),
        -int(parameters["hold_bars"]),
        str(parameters["name"]),
    )


def _prefix_record(cfg: Config, market: pd.DataFrame, cutoff: str) -> dict[str, Any]:
    columns = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "quote_asset_volume",
        "taker_buy_quote",
        "open_interest",
    ]
    digest = hashlib.sha256()
    hashed = pd.util.hash_pandas_object(market[columns], index=False).to_numpy(
        dtype="<u8"
    )
    digest.update(hashed.tobytes())
    return {
        "cutoff": cutoff,
        "rows": len(market),
        "first": str(market["date"].iloc[0]),
        "last": str(market["date"].iloc[-1]),
        "frame_hash": digest.hexdigest(),
        "market_file_sha256": _sha256(cfg.market_csv),
        "oi_file_sha256": _sha256(cfg.oi_csv),
    }


def _atomic_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def _render_pre2024(payload: dict[str, Any]) -> str:
    lines = [
        "# OFBR-1 pre-2024 selection",
        "",
        "Metric: `absolute return / full-calendar CAGR / strict MDD / CAGR-MDD / trades`.",
        "",
        f"- tested: {payload['tested']}",
        f"- passed frozen shortlist gate: {payload['passed']}",
        f"- status: **{payload['decision']}**",
        "",
        "| # | candidate | calibration | 2023 | 2023 H1 | 2023 H2 |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(payload["shortlist"], start=1):
        values = []
        for split in ("calibration", "train_2023", "train_2023_h1", "train_2023_h2"):
            metric = row["stats"][split]
            values.append(
                f"{metric['absolute_return_pct']:.2f}/{metric['cagr_pct']:.2f}/"
                f"{metric['strict_mdd_pct']:.2f}/"
                f"{metric['cagr_to_strict_mdd']:.2f}/{metric['trades']}"
            )
        lines.append(
            f"| {index} | `{row['candidate']['name']}` | "
            + " | ".join(values)
            + " |"
        )
    lines += [
        "",
        "- The returned source frame contains no row at or after `2024-01-01`.",
        "- OI is delayed one complete 5-minute bar and missing values fail closed.",
        "- No 2024+ outcome is opened in this artifact.",
    ]
    return "\n".join(lines) + "\n"


def run_pre2024(cfg: Config) -> dict[str, Any]:
    preregistration = load_preregistration(cfg.preregistration)
    market = load_sources(cfg, cutoff="2024-01-01")
    dates = pd.to_datetime(market["date"])
    features = build_features(market)
    thresholds = fit_thresholds(features, dates, preregistration)
    rows: list[dict[str, Any]] = []
    for candidate in candidate_grid(preregistration):
        signals = signal_positions(
            market, features, thresholds, candidate, preregistration
        )
        stats: dict[str, dict[str, Any]] = {}
        for window in (
            "calibration",
            "calibration_2020q4",
            "calibration_2021",
            "calibration_2022",
            "train_2023",
            "train_2023_h1",
            "train_2023_h2",
        ):
            start, end = WINDOWS[window]
            stats[window] = _public_metric(
                simulate(
                    market,
                    signals,
                    start=start,
                    end=end,
                    hold_bars=int(candidate["hold_bars"]),
                    leverage=cfg.leverage,
                    cost_rate=cfg.cost_rate,
                )
            )
        row = {
            "candidate": candidate,
            "stats": stats,
            "passes": _pre2024_passes(stats),
        }
        rows.append(row)
    passed = sorted(
        (row for row in rows if row["passes"]), key=_pre2024_key, reverse=True
    )
    shortlist = passed[:10]
    source_prefix = _prefix_record(cfg, market, "2024-01-01")
    freeze = {
        "preregistration_sha256": _sha256(cfg.preregistration),
        "source_prefix": source_prefix,
        "thresholds": thresholds,
        "shortlist": shortlist,
    }
    payload = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "phase": "pre2024_selection",
        "config": asdict(cfg),
        "preregistration": str(cfg.preregistration),
        "preregistration_sha256": freeze["preregistration_sha256"],
        "source_prefix": source_prefix,
        "thresholds": thresholds,
        "tested": len(rows),
        "passed": len(passed),
        "shortlist": shortlist,
        "decision": "open_test2024" if shortlist else "reject_pre2024",
        "test2024_opened": False,
        "future_opened": False,
        "freeze_hash": _json_hash(freeze),
        "all_pre2024_rows": rows,
    }
    _atomic_json(cfg.pre2024_output, payload)
    docs = Path(cfg.pre2024_docs)
    docs.parent.mkdir(parents=True, exist_ok=True)
    docs.write_text(_render_pre2024(payload), encoding="utf-8")
    return payload


def _validate_pre2024_manifest(
    cfg: Config, preregistration: dict[str, Any], payload: dict[str, Any]
) -> None:
    if payload.get("phase") != "pre2024_selection":
        raise RuntimeError("invalid pre-2024 manifest phase")
    if payload.get("test2024_opened") is not False:
        raise RuntimeError("test2024 has already been opened")
    if payload.get("decision") != "open_test2024" or not payload.get("shortlist"):
        raise RuntimeError("pre-2024 family was not admitted")
    if payload.get("preregistration_sha256") != _sha256(cfg.preregistration):
        raise RuntimeError("preregistration hash drifted")
    freeze = {
        "preregistration_sha256": payload["preregistration_sha256"],
        "source_prefix": payload["source_prefix"],
        "thresholds": payload["thresholds"],
        "shortlist": payload["shortlist"],
    }
    if payload.get("freeze_hash") != _json_hash(freeze):
        raise RuntimeError("pre-2024 freeze hash mismatch")
    if len(candidate_grid(preregistration)) != 96:
        raise RuntimeError("candidate grid drifted")


def _test2024_passes(stats: dict[str, dict[str, Any]]) -> bool:
    full = stats["test_2024"]
    return bool(
        full["absolute_return_pct"] > 0.0
        and full["cagr_to_strict_mdd"] >= 3.0
        and full["strict_mdd_pct"] <= 15.0
        and full["trades"] >= 12
    )


def _test2024_key(row: dict[str, Any]) -> tuple[Any, ...]:
    pre = row["pre2024_stats"]["train_2023"]
    test = row["test2024_stats"]["test_2024"]
    return (
        min(pre["cagr_to_strict_mdd"], test["cagr_to_strict_mdd"]),
        test["cagr_to_strict_mdd"],
        test["absolute_return_pct"],
        -test["strict_mdd_pct"],
        str(row["candidate"]["name"]),
    )


def _render_test2024(payload: dict[str, Any]) -> str:
    lines = [
        "# OFBR-1 frozen 2024 selection",
        "",
        "Metric: `absolute return / full-calendar CAGR / strict MDD / CAGR-MDD / trades`.",
        "",
        f"- candidates opened: {payload['opened_candidates']}",
        f"- standalone passers: {payload['standalone_passers']}",
        f"- status: **{payload['decision']}**",
        "",
        "| # | candidate | 2023 | 2024 | 2024 H1 | 2024 H2 |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(payload["ranked"], start=1):
        metrics = []
        for metric in (
            row["pre2024_stats"]["train_2023"],
            row["test2024_stats"]["test_2024"],
            row["test2024_stats"]["test_2024_h1"],
            row["test2024_stats"]["test_2024_h2"],
        ):
            metrics.append(
                f"{metric['absolute_return_pct']:.2f}/{metric['cagr_pct']:.2f}/"
                f"{metric['strict_mdd_pct']:.2f}/"
                f"{metric['cagr_to_strict_mdd']:.2f}/{metric['trades']}"
            )
        lines.append(
            f"| {index} | `{row['candidate']['name']}` | "
            + " | ".join(metrics)
            + " |"
        )
    lines += [
        "",
        "- The returned source frame contains no row at or after `2025-01-01`.",
        "- The unique top-1 is frozen before any 2025/2026 portfolio replay.",
        "- Standalone passage is necessary but not sufficient; Gross9 marginal contribution is the next gate.",
    ]
    return "\n".join(lines) + "\n"


def run_test2024(cfg: Config) -> dict[str, Any]:
    preregistration = load_preregistration(cfg.preregistration)
    pre_path = Path(cfg.pre2024_output)
    if not pre_path.exists():
        raise FileNotFoundError("pre-2024 manifest is required")
    pre = json.loads(pre_path.read_text(encoding="utf-8"))
    _validate_pre2024_manifest(cfg, preregistration, pre)

    # Burn the stage seal before opening the test source.
    pre["test2024_opened"] = True
    pre["test2024_opened_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(pre_path, pre)

    prefix = load_sources(cfg, cutoff="2024-01-01")
    if _prefix_record(cfg, prefix, "2024-01-01") != pre["source_prefix"]:
        raise RuntimeError("pre-2024 source prefix drifted")
    market = load_sources(cfg, cutoff="2025-01-01")
    features = build_features(market)
    rows: list[dict[str, Any]] = []
    for frozen in pre["shortlist"]:
        candidate = frozen["candidate"]
        signals = signal_positions(
            market, features, pre["thresholds"], candidate, preregistration
        )
        stats: dict[str, dict[str, Any]] = {}
        for window in ("test_2024", "test_2024_h1", "test_2024_h2"):
            start, end = WINDOWS[window]
            stats[window] = _public_metric(
                simulate(
                    market,
                    signals,
                    start=start,
                    end=end,
                    hold_bars=int(candidate["hold_bars"]),
                    leverage=cfg.leverage,
                    cost_rate=cfg.cost_rate,
                )
            )
        stress = _public_metric(
            simulate(
                market,
                signals,
                start=WINDOWS["test_2024"][0],
                end=WINDOWS["test_2024"][1],
                hold_bars=int(candidate["hold_bars"]),
                leverage=cfg.leverage,
                cost_rate=cfg.stress_cost_rate,
            )
        )
        row = {
            "candidate": candidate,
            "pre2024_stats": frozen["stats"],
            "test2024_stats": stats,
            "test2024_stress": stress,
            "standalone_passes": _test2024_passes(stats),
        }
        rows.append(row)
    ranked = sorted(rows, key=_test2024_key, reverse=True)
    passers = [row for row in ranked if row["standalone_passes"]]
    top1 = passers[0] if passers else None
    freeze = {
        "pre2024_freeze_hash": pre["freeze_hash"],
        "test_source_prefix": _prefix_record(cfg, market, "2025-01-01"),
        "top1": top1,
    }
    payload = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "phase": "test2024_selection",
        "pre2024_manifest": str(pre_path),
        "pre2024_freeze_hash": pre["freeze_hash"],
        "test_source_prefix": freeze["test_source_prefix"],
        "opened_candidates": len(rows),
        "standalone_passers": len(passers),
        "ranked": ranked,
        "top1": top1,
        "decision": "open_portfolio_gate" if top1 else "reject_test2024",
        "future_opened": False,
        "freeze_hash": _json_hash(freeze),
    }
    _atomic_json(cfg.test2024_output, payload)
    docs = Path(cfg.test2024_docs)
    docs.parent.mkdir(parents=True, exist_ok=True)
    docs.write_text(_render_test2024(payload), encoding="utf-8")
    return payload


def parse_args() -> tuple[str, Config]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pre2024", "test2024"))
    parser.add_argument("--market-csv", default=Config.market_csv)
    parser.add_argument("--oi-csv", default=Config.oi_csv)
    parser.add_argument("--preregistration", default=Config.preregistration)
    parser.add_argument("--pre2024-output", default=Config.pre2024_output)
    parser.add_argument("--pre2024-docs", default=Config.pre2024_docs)
    parser.add_argument("--test2024-output", default=Config.test2024_output)
    parser.add_argument("--test2024-docs", default=Config.test2024_docs)
    args = parser.parse_args()
    values = vars(args)
    stage = values.pop("stage")
    return stage, Config(**values)


def main() -> None:
    stage, cfg = parse_args()
    payload = run_pre2024(cfg) if stage == "pre2024" else run_test2024(cfg)
    print(
        json.dumps(
            {
                "phase": payload["phase"],
                "decision": payload["decision"],
                "tested": payload.get("tested", payload.get("opened_candidates")),
                "passed": payload.get("passed", payload.get("standalone_passers")),
                "output": (
                    cfg.pre2024_output if stage == "pre2024" else cfg.test2024_output
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
