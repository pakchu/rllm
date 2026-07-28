"""Stage-gated search for the preregistered REX-short disagreement rebound.

RSDR-1 recomputes a raw, causal REX pullback/reclaim clock directly from
completed market bars.  It intentionally does not load the future-labelled REX
SFT JSONL.  A candidate waits after a raw REX SHORT anchor and enters a separate
LONG sleeve only when delayed OI and completed price/flow bars indicate failed
downside continuation.

The first stage in this module is deliberately prefix-only: ``pre2024`` returns
no market or OI row at or after 2024-01-01.  It fits thresholds on 2020-09
through 2021-12 and evaluates support only in 2022 and 2023.  The 2024
portfolio-marginal stage is added only if this family survives the frozen
pre-2024 gate.
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

from training.build_rex_event_reasoning_policy_data import (
    _build_light_rex_features,
    _rex_pullback_reclaim_arrays,
)
from training.portfolio_opt_added_alpha_update import favorable_path
from training.portfolio_opt_new_alpha_pool import _event_path


PREREGISTRATION = Path(
    "results/rex_short_disagreement_rebound_preregistration_2026-07-28.json"
)
PRE2024_OUTPUT = Path(
    "results/rex_short_disagreement_rebound_pre2024_selection_2026-07-28.json"
)
PRE2024_DOC = Path(
    "docs/rex-short-disagreement-rebound-pre2024-selection-2026-07-28.md"
)

WINDOWS = {
    "fit": ("2020-09-01", "2022-01-01"),
    "support_2022": ("2022-01-01", "2023-01-01"),
    "support_2022_h1": ("2022-01-01", "2022-07-01"),
    "support_2022_h2": ("2022-07-01", "2023-01-01"),
    "robustness_2023": ("2023-01-01", "2024-01-01"),
    "robustness_2023_h1": ("2023-01-01", "2023-07-01"),
    "robustness_2023_h2": ("2023-07-01", "2024-01-01"),
    "combined_2022_2023": ("2022-01-01", "2024-01-01"),
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
    leverage: float = 0.5
    cost_rate: float = 0.0006


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
    if payload.get("name") != "rex_short_disagreement_rebound":
        raise RuntimeError("unexpected preregistration name")
    if payload.get("status") != "preregistered_before_outcome_scan":
        raise RuntimeError("preregistration is not outcome-blind")
    if payload["frozen_grid"].get("mechanism_candidate_count") != 48:
        raise RuntimeError("frozen mechanism count drifted")
    if payload["causality_contract"].get("future_can_rerank") is not False:
        raise RuntimeError("future reranking is not disabled")
    return payload


def _read_market_prefix(path: str | Path, cutoff: str) -> pd.DataFrame:
    columns = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "taker_buy_base",
        "taker_buy_quote",
    ]
    limit = pd.Timestamp(cutoff)
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        usecols=columns,
        parse_dates=["date"],
        compression="infer",
        chunksize=100_000,
    ):
        dates = pd.to_datetime(chunk["date"], utc=True, errors="raise").dt.tz_convert(
            None
        )
        chunk = chunk.assign(date=dates)
        keep = chunk["date"] < limit
        if keep.any():
            chunks.append(chunk.loc[keep])
        if (~keep).any() or (len(chunk) and chunk["date"].iloc[-1] >= limit):
            break
    if not chunks:
        raise RuntimeError("market prefix is empty")
    return (
        pd.concat(chunks, ignore_index=True)
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )


def _read_oi_prefix(path: str | Path, cutoff: str) -> pd.DataFrame:
    limit = pd.Timestamp(cutoff)
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path, usecols=["date", "open_interest"], chunksize=100_000
    ):
        dates = pd.to_datetime(chunk["date"], utc=True, errors="raise").dt.tz_convert(
            None
        )
        chunk = chunk.assign(date=dates)
        keep = chunk["date"] < limit
        if keep.any():
            chunks.append(chunk.loc[keep])
        if (~keep).any() or (len(chunk) and chunk["date"].iloc[-1] >= limit):
            break
    if not chunks:
        raise RuntimeError("OI prefix is empty")
    return (
        pd.concat(chunks, ignore_index=True)
        .sort_values("date")
        .drop_duplicates("date", keep="last")
    )


def load_sources(cfg: Config, *, cutoff: str) -> pd.DataFrame:
    market = _read_market_prefix(cfg.market_csv, cutoff)
    oi = _read_oi_prefix(cfg.oi_csv, cutoff)
    merged = market.merge(oi, on="date", how="left", validate="one_to_one")
    if len(merged) and pd.Timestamp(merged["date"].max()) >= pd.Timestamp(cutoff):
        raise RuntimeError("source cutoff was not enforced")
    return merged.reset_index(drop=True)


def build_features(market: pd.DataFrame) -> pd.DataFrame:
    """Build completed-bar REX/OI/flow features; OI is delayed exactly one bar."""
    rex = _build_light_rex_features(market)
    strength, direction = _rex_pullback_reclaim_arrays(rex)

    close = pd.to_numeric(market["close"], errors="coerce")
    high = pd.to_numeric(market["high"], errors="coerce")
    low = pd.to_numeric(market["low"], errors="coerce")
    quote = pd.to_numeric(market["quote_asset_volume"], errors="coerce")
    taker_buy = pd.to_numeric(market["taker_buy_quote"], errors="coerce")
    observed_oi = pd.to_numeric(market["open_interest"], errors="coerce").shift(1)

    log_close = np.log(close.where(close > 0.0))
    log_oi = np.log(observed_oi.where(observed_oi > 0.0))
    price_ret_4h = log_close - log_close.shift(48)
    oi_ret_4h = log_oi - log_oi.shift(48)
    imbalance = (
        2.0 * taker_buy / quote.replace(0.0, np.nan) - 1.0
    ).clip(-1.0, 1.0)
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
            "rex_strength": strength,
            "rex_direction": direction,
            "price_ret_4h": price_ret_4h,
            "oi_ret_4h": oi_ret_4h,
            "oi_minus_price_4h": oi_ret_4h - price_ret_4h,
            "flow_1h": imbalance.rolling(12, min_periods=12).mean(),
            "flow_15m": imbalance.rolling(3, min_periods=3).mean(),
            "observed_oi": observed_oi,
            "oi_available": observed_oi.notna(),
            "atr_1h": true_range.rolling(12, min_periods=12).mean(),
        },
        index=market.index,
    ).replace([np.inf, -np.inf], np.nan)


def _sampled_positions(length: int, stride: int) -> np.ndarray:
    return np.arange(143, max(143, length), int(stride), dtype=np.int64)


def raw_short_anchors(
    features: pd.DataFrame,
    *,
    strength_threshold: float,
    stride_bars: int,
) -> np.ndarray:
    positions = _sampled_positions(len(features), stride_bars)
    strength = features["rex_strength"].to_numpy(float)
    direction = features["rex_direction"].to_numpy(float)
    active = (
        np.isfinite(strength[positions])
        & np.isfinite(direction[positions])
        & (strength[positions] > float(strength_threshold))
        & (direction[positions] < 0.0)
    )
    return positions[active]


def fit_thresholds(
    features: pd.DataFrame,
    dates: pd.Series,
    preregistration: dict[str, Any],
) -> dict[str, Any]:
    grid = preregistration["frozen_grid"]
    start, end = preregistration["causality_contract"]["rex_threshold_fit_window"]
    fit_mask = np.asarray(
        (dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end)), dtype=bool
    )
    strength = features["rex_strength"].to_numpy(float)
    positive = strength[fit_mask & np.isfinite(strength) & (strength > 0.0)]
    if len(positive) < 10_000:
        raise RuntimeError(f"insufficient REX threshold support: {len(positive)}")
    rex_threshold = float(
        np.quantile(positive, float(grid["rex_strength_quantile"]))
    )
    anchors = raw_short_anchors(
        features,
        strength_threshold=rex_threshold,
        stride_bars=int(grid["rex_stride_bars"]),
    )
    anchors = anchors[fit_mask[anchors]]
    divergence = features["oi_minus_price_4h"].to_numpy(float)
    finite = anchors[np.isfinite(divergence[anchors])]
    if len(finite) < 20:
        raise RuntimeError(f"insufficient REX-short OI support: {len(finite)}")
    return {
        "rex_strength": rex_threshold,
        "rex_positive_fit_rows": int(len(positive)),
        "rex_short_fit_anchors": int(len(anchors)),
        "rex_short_oi_fit_anchors": int(len(finite)),
        "oi_minus_price": {
            str(q): float(np.quantile(divergence[finite], float(q)))
            for q in grid["oi_minus_price_anchor_quantile"]
        },
    }


def candidate_grid(preregistration: dict[str, Any]) -> list[dict[str, Any]]:
    grid = preregistration["frozen_grid"]
    keys = (
        "oi_minus_price_anchor_quantile",
        "confirmation_wait_bars",
        "minimum_oi_retention",
        "minimum_post_anchor_range_position",
        "hold_bars",
    )
    rows: list[dict[str, Any]] = []
    for values in itertools.product(*(grid[key] for key in keys)):
        row = dict(zip(keys, values))
        row["name"] = (
            "rsdr"
            f"_d{int(round(float(row['oi_minus_price_anchor_quantile']) * 100))}"
            f"_w{int(row['confirmation_wait_bars'])}"
            f"_oi{int(round(float(row['minimum_oi_retention']) * 10000))}"
            f"_r{int(round(float(row['minimum_post_anchor_range_position']) * 100))}"
            f"_h{int(row['hold_bars'])}"
        )
        rows.append(row)
    if len(rows) != int(grid["mechanism_candidate_count"]):
        raise RuntimeError("candidate grid size drifted")
    return rows


def signal_positions(
    market: pd.DataFrame,
    features: pd.DataFrame,
    thresholds: dict[str, Any],
    candidate: dict[str, Any],
    preregistration: dict[str, Any],
) -> np.ndarray:
    """Return completed confirmation-bar positions for one frozen candidate."""
    grid = preregistration["frozen_grid"]
    anchors = raw_short_anchors(
        features,
        strength_threshold=float(thresholds["rex_strength"]),
        stride_bars=int(grid["rex_stride_bars"]),
    )
    divergence = features["oi_minus_price_4h"].to_numpy(float)
    slow_flow = features["flow_1h"].to_numpy(float)
    fast_flow = features["flow_15m"].to_numpy(float)
    observed_oi = features["observed_oi"].to_numpy(float)
    atr = features["atr_1h"].to_numpy(float)
    close = pd.to_numeric(market["close"], errors="coerce").to_numpy(float)
    high = pd.to_numeric(market["high"], errors="coerce").to_numpy(float)
    low = pd.to_numeric(market["low"], errors="coerce").to_numpy(float)

    quantile = str(candidate["oi_minus_price_anchor_quantile"])
    anchors = anchors[
        np.isfinite(divergence[anchors])
        & np.isfinite(slow_flow[anchors])
        & np.isfinite(observed_oi[anchors])
        & np.isfinite(atr[anchors])
        & (divergence[anchors] >= float(thresholds["oi_minus_price"][quantile]))
    ]
    wait = int(candidate["confirmation_wait_bars"])
    retention_floor = float(candidate["minimum_oi_retention"])
    recovery_floor = float(candidate["minimum_post_anchor_range_position"])
    atr_multiple = float(grid["post_anchor_atr_multiple"])
    confirmations: list[int] = []
    for raw_anchor in anchors:
        anchor = int(raw_anchor)
        confirmation = anchor + wait
        if confirmation >= len(market):
            continue
        values = (
            close[anchor],
            close[confirmation],
            slow_flow[anchor],
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
        post_position = (float(close[confirmation]) - post_low) / (
            post_high - post_low
        )
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


def _strict_metric(
    returns: np.ndarray,
    market_low: np.ndarray,
    market_high: np.ndarray,
    *,
    years: float,
    trades: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(returns):
        equity_after = np.cumprod(np.maximum(0.0, 1.0 + returns))
        equity_before = np.r_[1.0, equity_after[:-1]]
        low_value = equity_before * np.maximum(0.0, 1.0 + market_low)
        high_value = equity_before * np.maximum(0.0, 1.0 + market_high)
        upper = np.maximum.reduce(
            [equity_before, equity_after, low_value, high_value]
        )
        lower = np.minimum.reduce(
            [equity_before, equity_after, low_value, high_value]
        )
        peak = np.maximum.accumulate(upper)
        mdd = float(np.max(1.0 - lower / np.maximum(peak, 1e-12))) * 100.0
        final = float(equity_after[-1])
    else:
        mdd = 0.0
        final = 1.0
    absolute_return = (final - 1.0) * 100.0
    cagr = (final ** (1.0 / years) - 1.0) * 100.0 if final > 0.0 else -100.0
    realized = np.asarray([float(row["net_return"]) for row in trades], dtype=float)
    if len(realized) > 1 and float(np.std(realized, ddof=1)) > 0.0:
        t_stat = float(
            np.mean(realized) / (np.std(realized, ddof=1) / math.sqrt(len(realized)))
        )
    else:
        t_stat = 0.0
    wins = int(np.sum(realized > 0.0))
    return {
        "absolute_return_pct": absolute_return,
        "cagr_pct": cagr,
        "strict_mdd_pct": mdd,
        "cagr_to_strict_mdd": cagr / mdd if mdd > 1e-12 else 0.0,
        "trades": len(trades),
        "wins": wins,
        "win_rate": wins / len(trades) if trades else 0.0,
        "mean_trade_bps": float(realized.mean() * 10_000.0) if len(realized) else 0.0,
        "trade_t_stat": t_stat,
        "signal_hash": hashlib.sha256(
            np.asarray(
                [int(row["signal_position"]) for row in trades], dtype="<i8"
            ).tobytes()
        ).hexdigest(),
    }


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
    """Replay the long sleeve with the repository's exact fixed-hold path."""
    dates = pd.to_datetime(market["date"])
    mask = np.asarray(
        (dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end)), dtype=bool
    )
    positions = np.flatnonzero(mask)
    if not len(positions):
        raise RuntimeError(f"empty simulation window: {start}..{end}")
    first, last = int(positions[0]), int(positions[-1]) + 1
    returns = np.zeros(len(market), dtype=np.float64)
    market_low = np.zeros(len(market), dtype=np.float64)
    market_high = np.zeros(len(market), dtype=np.float64)
    trades: list[dict[str, Any]] = []
    next_allowed = first
    for raw_signal in sorted(set(int(value) for value in signals)):
        signal = int(raw_signal)
        if signal < first or signal >= last or signal < next_allowed:
            continue
        path = _event_path(
            market,
            signal,
            side="long",
            hold=int(hold_bars),
            cost_rate=float(cost_rate),
            entry_delay=1,
            leverage=float(leverage),
        )
        if path is None:
            continue
        event_return, event_adverse, realized = path
        nonzero = np.flatnonzero(np.abs(event_return) > 1e-15)
        if not len(nonzero):
            continue
        exit_position = int(nonzero[-1])
        if exit_position >= last or not mask[exit_position]:
            continue
        event_favorable = favorable_path(
            market,
            signal_position=signal,
            exit_position=exit_position,
            side="long",
            leverage=float(leverage),
        )
        returns += event_return
        market_low += event_adverse
        market_high += event_favorable
        trades.append(
            {
                "signal_position": signal,
                "signal_date": str(dates.iloc[signal]),
                "entry_date": str(dates.iloc[signal + 1]),
                "exit_date": str(dates.iloc[exit_position]),
                "side": "LONG",
                "net_return": float(realized),
            }
        )
        next_allowed = exit_position + 1
    local = slice(first, last)
    return _strict_metric(
        returns[local],
        market_low[local],
        market_high[local],
        years=_years(start, end),
        trades=trades,
    )


def _pre2024_passes(stats: dict[str, dict[str, Any]]) -> bool:
    support = stats["support_2022"]
    robustness = stats["robustness_2023"]
    combined = stats["combined_2022_2023"]
    return bool(
        support["trades"] >= 4
        and robustness["trades"] >= 4
        and combined["trades"] >= 12
        and combined["absolute_return_pct"] > 0.0
        and support["absolute_return_pct"] >= -1.0
        and robustness["absolute_return_pct"] >= -1.0
        and max(
            support["strict_mdd_pct"],
            robustness["strict_mdd_pct"],
            combined["strict_mdd_pct"],
        )
        <= 8.0
    )


def _pre2024_key(row: dict[str, Any]) -> tuple[Any, ...]:
    stats = row["stats"]
    support = stats["support_2022"]
    robustness = stats["robustness_2023"]
    combined = stats["combined_2022_2023"]
    candidate = row["candidate"]
    return (
        combined["cagr_to_strict_mdd"],
        min(support["absolute_return_pct"], robustness["absolute_return_pct"]),
        combined["absolute_return_pct"],
        -int(candidate["hold_bars"]),
        str(candidate["name"]),
    )


def _prefix_record(cfg: Config, market: pd.DataFrame, cutoff: str) -> dict[str, Any]:
    columns = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "taker_buy_base",
        "taker_buy_quote",
        "open_interest",
    ]
    hashed = pd.util.hash_pandas_object(market[columns], index=False).to_numpy(
        dtype="<u8"
    )
    return {
        "cutoff": cutoff,
        "rows": len(market),
        "first": str(market["date"].iloc[0]),
        "last": str(market["date"].iloc[-1]),
        "frame_hash": hashlib.sha256(hashed.tobytes()).hexdigest(),
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


def _metric_cell(metric: dict[str, Any]) -> str:
    return (
        f"{metric['absolute_return_pct']:.2f}/{metric['cagr_pct']:.2f}/"
        f"{metric['strict_mdd_pct']:.2f}/"
        f"{metric['cagr_to_strict_mdd']:.2f}/{metric['trades']}"
    )


def _render_pre2024(payload: dict[str, Any]) -> str:
    displayed = payload["shortlist"] or payload["diagnostic_top"]
    table_label = (
        "frozen shortlist"
        if payload["shortlist"]
        else "top failed cells (diagnostic only; no cell admitted)"
    )
    lines = [
        "# RSDR-1 pre-2024 support selection",
        "",
        "Metric: `absolute return / full-calendar CAGR / strict MDD / CAGR-MDD / trades`.",
        "",
        f"- raw fit-window REX-short anchors: {payload['thresholds']['rex_short_fit_anchors']}",
        f"- raw fit-window anchors with delayed OI: {payload['thresholds']['rex_short_oi_fit_anchors']}",
        f"- raw 2022 REX-short anchors: {payload['raw_anchor_counts']['support_2022']}",
        f"- raw 2023 REX-short anchors: {payload['raw_anchor_counts']['robustness_2023']}",
        f"- tested: {payload['tested']}",
        f"- passed frozen shortlist gate: {payload['passed']}",
        f"- status: **{payload['decision']}**",
        f"- table: {table_label}",
        "",
        "| # | candidate | pass | 2022 | 2022 H1 | 2022 H2 | 2023 | 2023 H1 | 2023 H2 | combined |",
        "|---:|---|:---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(displayed, start=1):
        stats = row["stats"]
        cells = [
            _metric_cell(stats[name])
            for name in (
                "support_2022",
                "support_2022_h1",
                "support_2022_h2",
                "robustness_2023",
                "robustness_2023_h1",
                "robustness_2023_h2",
                "combined_2022_2023",
            )
        ]
        lines.append(
            f"| {index} | `{row['candidate']['name']}` | "
            f"{'Y' if row['passes'] else 'N'} | "
            + " | ".join(cells)
            + " |"
        )
    lines += [
        "",
        "- The returned source frame contains no row at or after `2024-01-01`.",
        "- Raw REX strength/direction is recomputed from completed market bars.",
        "- The future-labelled REX SFT JSONL is never loaded.",
        "- OI is delayed exactly one complete 5-minute bar and missing values fail closed.",
        "- No 2024+ outcome or Gross9 marginal replay is opened in this artifact.",
    ]
    return "\n".join(lines) + "\n"


def run_pre2024(cfg: Config) -> dict[str, Any]:
    preregistration = load_preregistration(cfg.preregistration)
    market = load_sources(cfg, cutoff="2024-01-01")
    dates = pd.to_datetime(market["date"])
    features = build_features(market)
    thresholds = fit_thresholds(features, dates, preregistration)
    raw_anchors = raw_short_anchors(
        features,
        strength_threshold=float(thresholds["rex_strength"]),
        stride_bars=int(preregistration["frozen_grid"]["rex_stride_bars"]),
    )
    raw_anchor_counts = {
        name: int(
            np.sum(
                (dates.iloc[raw_anchors] >= pd.Timestamp(start))
                & (dates.iloc[raw_anchors] < pd.Timestamp(end))
            )
        )
        for name, (start, end) in {
            "fit": WINDOWS["fit"],
            "support_2022": WINDOWS["support_2022"],
            "robustness_2023": WINDOWS["robustness_2023"],
        }.items()
    }
    rows: list[dict[str, Any]] = []
    metric_windows = (
        "support_2022",
        "support_2022_h1",
        "support_2022_h2",
        "robustness_2023",
        "robustness_2023_h1",
        "robustness_2023_h2",
        "combined_2022_2023",
    )
    for candidate in candidate_grid(preregistration):
        signals = signal_positions(
            market, features, thresholds, candidate, preregistration
        )
        stats: dict[str, dict[str, Any]] = {}
        for window in metric_windows:
            start, end = WINDOWS[window]
            stats[window] = simulate(
                market,
                signals,
                start=start,
                end=end,
                hold_bars=int(candidate["hold_bars"]),
                leverage=cfg.leverage,
                cost_rate=cfg.cost_rate,
            )
        rows.append(
            {
                "candidate": candidate,
                "stats": stats,
                "passes": _pre2024_passes(stats),
            }
        )
    passed = sorted(
        (row for row in rows if row["passes"]), key=_pre2024_key, reverse=True
    )
    max_candidates = int(
        preregistration["selection_contract"]["pre2024_shortlist"]["max_candidates"]
    )
    shortlist = passed[:max_candidates]
    diagnostic_top = sorted(rows, key=_pre2024_key, reverse=True)[:max_candidates]
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
        "raw_anchor_counts": raw_anchor_counts,
        "tested": len(rows),
        "passed": len(passed),
        "shortlist": shortlist,
        "diagnostic_top": diagnostic_top,
        "decision": "open_test2024" if shortlist else "reject_pre2024",
        "test2024_opened": False,
        "future_opened": False,
        "future_labelled_rex_jsonl_opened": False,
        "freeze_hash": _json_hash(freeze),
        "all_pre2024_rows": rows,
    }
    _atomic_json(cfg.pre2024_output, payload)
    destination = Path(cfg.pre2024_docs)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_render_pre2024(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("pre2024",))
    parser.add_argument("--market-csv", default=Config.market_csv)
    parser.add_argument("--oi-csv", default=Config.oi_csv)
    parser.add_argument("--preregistration", default=Config.preregistration)
    parser.add_argument("--pre2024-output", default=Config.pre2024_output)
    parser.add_argument("--pre2024-docs", default=Config.pre2024_docs)
    args = parser.parse_args()
    cfg = Config(
        market_csv=args.market_csv,
        oi_csv=args.oi_csv,
        preregistration=args.preregistration,
        pre2024_output=args.pre2024_output,
        pre2024_docs=args.pre2024_docs,
    )
    payload = run_pre2024(cfg)
    print(
        json.dumps(
            {
                "phase": payload["phase"],
                "tested": payload["tested"],
                "passed": payload["passed"],
                "decision": payload["decision"],
                "output": cfg.pre2024_output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
