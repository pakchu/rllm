"""Search a causal BTC spot/perpetual delta-neutral funding-carry sleeve.

This research family is intentionally different from the repository's
directional BTC alphas: it holds equal-dollar long Binance spot and short
Binance USD-M perpetual exposure, then harvests realized positive funding.
Selection is physically sealed before 2024.  The script never reads an OOS
source and writes a standalone frozen-policy manifest for a later one-shot
2024+ evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SELECTION_END = "2024-01-01"
DEFAULT_FUTURES = (
    "data/binance_perp_btc_1m_2020_2023.csv.gz"
)
DEFAULT_SPOT = "data/binance_spot_btc_1m_2020_2023.csv.gz"
DEFAULT_FUNDING = "data/binance_funding_btc_2020_2023.csv.gz"
DEFAULT_SOURCE_MANIFEST = (
    "results/delta_neutral_carry_sources_pre2024_manifest_2026-07-16.json"
)
DEFAULT_OUTPUT = "results/delta_neutral_funding_carry_pre2024_2026-07-16.json"
DEFAULT_POLICY_MANIFEST = (
    "results/delta_neutral_funding_carry_frozen_policy_2026-07-16.json"
)
DEFAULT_DOCS = "docs/delta-neutral-funding-carry-pre2024-2026-07-16.md"

WINDOWS: dict[str, tuple[str, str]] = {
    "fit_2020h1": ("2020-01-01", "2020-07-01"),
    "fit_2020h2": ("2020-07-01", "2021-01-01"),
    "fit_2021h1": ("2021-01-01", "2021-07-01"),
    "fit_2021h2": ("2021-07-01", "2022-01-01"),
    "fit_2022h1": ("2022-01-01", "2022-07-01"),
    "fit_2022h2": ("2022-07-01", "2023-01-01"),
    "fit_2020_2022": ("2020-01-01", "2023-01-01"),
    "select_2023h1": ("2023-01-01", "2023-07-01"),
    "select_2023h2": ("2023-07-01", SELECTION_END),
    "select_2023": ("2023-01-01", SELECTION_END),
}
HALF_WINDOWS = tuple(name for name in WINDOWS if name.endswith(("h1", "h2")))


@dataclass(frozen=True)
class Config:
    futures_csv: str = DEFAULT_FUTURES
    spot_csv: str = DEFAULT_SPOT
    funding_csv: str = DEFAULT_FUNDING
    source_manifest: str = DEFAULT_SOURCE_MANIFEST
    output: str = DEFAULT_OUTPUT
    policy_manifest: str = DEFAULT_POLICY_MANIFEST
    docs_output: str = DEFAULT_DOCS
    cutoff: str = SELECTION_END
    gross_exposure: float = 1.0
    spot_fee_rate: float = 0.0010
    perp_fee_rate: float = 0.0005
    spot_slippage_rate: float = 0.0001
    perp_slippage_rate: float = 0.0001
    incomplete_spot_cushion: float = 0.0025
    bootstrap_samples: int = 5_000
    bootstrap_seed: int = 271_828


@dataclass(frozen=True, order=True)
class Policy:
    lookback_events: int
    entry_threshold: float
    exit_threshold: float
    min_hold_events: int


@dataclass(frozen=True)
class CostModel:
    spot_rate: float
    perp_rate: float


@dataclass
class Sources:
    market: pd.DataFrame
    funding: pd.DataFrame
    source_hashes: dict[str, str]
    diagnostics: dict[str, Any]


def resolve_existing(path: str) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate.resolve()
    fallback = Path("/home/pakchu/rllm") / path
    if fallback.exists():
        return fallback.resolve()
    raise FileNotFoundError(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _dates(frame: pd.DataFrame, column: str = "date") -> pd.Series:
    return pd.to_datetime(frame[column], utc=True, errors="raise", format="mixed").dt.tz_convert(None)


def _validate_ohlc(frame: pd.DataFrame, prefix: str = "") -> None:
    names = [f"{prefix}{name}" for name in ("open", "high", "low", "close")]
    values = frame[names].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(values.to_numpy(float)).all() or (values <= 0.0).any().any():
        raise ValueError(f"{prefix or 'market '}OHLC must be positive and finite")
    if (values[names[1]] < values[[names[0], names[2], names[3]]].max(axis=1)).any():
        raise ValueError(f"{prefix or 'market '}high is inconsistent")
    if (values[names[2]] > values[[names[0], names[1], names[3]]].min(axis=1)).any():
        raise ValueError(f"{prefix or 'market '}low is inconsistent")


def _max_true_streak(mask: np.ndarray) -> int:
    longest = current = 0
    for value in mask:
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return int(longest)


def reconstruct_spot(
    futures: pd.DataFrame,
    spot: pd.DataFrame,
    *,
    cushion: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join spot to the complete futures clock and conservatively repair gaps.

    Only present one-minute spot bars update the causal spot/perp basis.
    Missing bars use the last complete basis and futures OHLC.  Their
    high/low are widened by ``cushion`` and include every actually observed
    spot point, inflating rather than suppressing strict intrabar drawdown.
    """
    if not 0.0 <= cushion < 0.05:
        raise ValueError("incomplete spot cushion must be in [0, 0.05)")
    required = {"date", "open", "high", "low", "close"}
    if not required.issubset(spot.columns):
        raise ValueError(f"spot source missing columns: {sorted(required - set(spot.columns))}")
    if spot["date"].duplicated().any():
        raise ValueError("spot source contains duplicate one-minute timestamps")

    indexed = spot.set_index("date").reindex(futures["date"])
    observed_fields = indexed[["open", "high", "low", "close"]].notna()
    full = observed_fields.all(axis=1).to_numpy(bool)
    partial = (observed_fields.any(axis=1) & ~observed_fields.all(axis=1)).to_numpy(bool)
    missing = (~observed_fields.any(axis=1)).to_numpy(bool)
    perp_close = futures["close"].to_numpy(float)
    raw_close = pd.to_numeric(indexed["close"], errors="coerce").to_numpy(float)
    complete_basis = np.where(full, raw_close / perp_close - 1.0, np.nan)
    # shift before ffill: a repaired bar may use only a basis known before its open.
    prior_basis = pd.Series(complete_basis).shift(1).ffill().to_numpy(float)
    repaired = partial | missing
    if repaired.any() and not np.isfinite(prior_basis[repaired]).all():
        raise ValueError("cannot repair spot bars before a complete causal basis exists")

    output = futures[["date"]].copy()
    observed = {
        name: pd.to_numeric(indexed[name], errors="coerce").to_numpy(float)
        for name in ("open", "high", "low", "close")
    }
    for name in ("open", "high", "low", "close"):
        output[f"spot_{name}"] = observed[name]

    p_open = futures["open"].to_numpy(float)
    p_high = futures["high"].to_numpy(float)
    p_low = futures["low"].to_numpy(float)
    proxy_open = p_open * (1.0 + prior_basis)
    proxy_close = perp_close * (1.0 + prior_basis)
    proxy_high = p_high * (1.0 + prior_basis + cushion)
    proxy_low = p_low * (1.0 + prior_basis - cushion)
    if repaired.any():
        output.loc[repaired, "spot_open"] = proxy_open[repaired]
        output.loc[repaired, "spot_close"] = proxy_close[repaired]
        observed_points = pd.DataFrame(
            np.column_stack([observed[name] for name in observed])
        )
        observed_high = observed_points.max(axis=1, skipna=True).to_numpy(float)
        observed_low = observed_points.min(axis=1, skipna=True).to_numpy(float)
        # All-NaN missing rows remain NaN, then fmax/fmin select the finite proxy.
        widened_high = np.fmax(proxy_high, observed_high)
        widened_low = np.fmin(proxy_low, observed_low)
        output.loc[repaired, "spot_high"] = widened_high[repaired]
        output.loc[repaired, "spot_low"] = widened_low[repaired]
    output["spot_proxy"] = repaired
    output["spot_observations"] = observed_fields.sum(axis=1).astype(int).to_numpy()
    _validate_ohlc(output, "spot_")
    diagnostics = {
        "complete_spot_bars": int(full.sum()),
        "partial_spot_bars": int(partial.sum()),
        "missing_spot_bars": int(missing.sum()),
        "proxy_spot_bars": int(repaired.sum()),
        "max_consecutive_proxy_bars": _max_true_streak(repaired),
        "proxy_rule": (
            "last complete prior one-minute spot/perp close basis; futures OHLC; high/low widened "
            f"by {cushion:.6f}; observed partial points retained in extrema"
        ),
    }
    return output, diagnostics


def map_funding_to_market(
    market: pd.DataFrame,
    funding: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Map settled funding to its next tradable open and a causal mark proxy."""
    output = funding.copy()
    bar_ns = market["date"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    event_ns = output["date"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    execution_mask = (
        (market["date"].dt.minute.to_numpy() % 5 == 0)
        & (market["date"].dt.second.to_numpy() == 0)
    )
    execution_indices = np.flatnonzero(execution_mask)
    execution_ns = bar_ns[execution_indices]
    slots = np.searchsorted(execution_ns, event_ns, side="right")
    exec_index = np.full(len(output), len(market), dtype=int)
    valid_slots = slots < len(execution_indices)
    exec_index[valid_slots] = execution_indices[slots[valid_slots]]
    output["exec_index"] = exec_index
    output["settlement_index"] = np.searchsorted(bar_ns, event_ns, side="left")
    one_minute_ns = int(pd.Timedelta("1min").value)
    fallback_index = np.searchsorted(bar_ns + one_minute_ns, event_ns, side="right") - 1
    fallback_mark = np.full(len(output), np.nan, dtype=float)
    valid_fallback = fallback_index >= 0
    fallback_mark[valid_fallback] = market["perp_close"].to_numpy(float)[
        fallback_index[valid_fallback]
    ]
    reported = output["reported_mark_price"].to_numpy(float)
    use_reported = np.isfinite(reported) & (reported > 0.0)
    output["settlement_mark"] = np.where(use_reported, reported, fallback_mark)
    output["settlement_mark_is_reported"] = use_reported
    output["fallback_index"] = fallback_index
    compare = np.isfinite(reported) & np.isfinite(fallback_mark) & (reported > 0.0)
    mark_error_bps = np.abs(reported[compare] / fallback_mark[compare] - 1.0) * 10_000.0
    diagnostics = {
        "funding_missing_reported_mark": int((~np.isfinite(reported) | (reported <= 0.0)).sum()),
        "funding_without_causal_fallback": int((~np.isfinite(fallback_mark)).sum()),
        "funding_settlement_marks_reported": int(use_reported.sum()),
        "funding_settlement_marks_fallback": int((~use_reported).sum()),
        "reported_vs_causal_mark_comparisons": int(compare.sum()),
        "reported_vs_causal_mark_median_abs_error_bps": (
            float(np.median(mark_error_bps)) if len(mark_error_bps) else None
        ),
        "reported_vs_causal_mark_p99_abs_error_bps": (
            float(np.quantile(mark_error_bps, 0.99)) if len(mark_error_bps) else None
        ),
        "funding_mark_policy": (
            "use actual reported funding mark when finite and positive; otherwise use the last "
            "fully completed USD-M one-minute close whose end <= funding_time"
        ),
    }
    return output, diagnostics


def load_sources(cfg: Config) -> Sources:
    if cfg.cutoff != SELECTION_END:
        raise ValueError("pre-2024 search cutoff is immutable")
    futures_path = resolve_existing(cfg.futures_csv)
    spot_path = resolve_existing(cfg.spot_csv)
    funding_path = resolve_existing(cfg.funding_csv)
    source_manifest_path = resolve_existing(cfg.source_manifest)

    source_manifest = json.loads(source_manifest_path.read_text())
    hashes = {
        "futures": _sha256(futures_path),
        "spot": _sha256(spot_path),
        "funding": _sha256(funding_path),
        "source_manifest": _sha256(source_manifest_path),
    }
    for name in ("futures", "spot", "funding"):
        manifest_name = "perp" if name == "futures" else name
        if hashes[name] != source_manifest["outputs"][manifest_name]["sha256"]:
            raise RuntimeError(f"{name} source hash does not match the frozen export manifest")
    if bool(source_manifest.get("database_snapshot_is_point_in_time", True)):
        raise RuntimeError("source manifest lost its explicit non-point-in-time disclosure")

    futures = pd.read_csv(futures_path, compression="infer")
    needed = {"date", "open", "high", "low", "close"}
    if not needed.issubset(futures.columns):
        raise ValueError(f"futures source missing columns: {sorted(needed - set(futures.columns))}")
    futures = futures[list(needed)].copy()
    futures["date"] = _dates(futures)
    futures = futures.sort_values("date").reset_index(drop=True)
    if futures["date"].duplicated().any() or not futures["date"].is_monotonic_increasing:
        raise ValueError("futures timestamps must be unique and increasing")
    expected = pd.date_range("2020-01-01", SELECTION_END, inclusive="left", freq="1min")
    if len(futures) != len(expected) or not np.array_equal(
        futures["date"].to_numpy(dtype="datetime64[ns]"), expected.to_numpy(dtype="datetime64[ns]")
    ):
        raise RuntimeError("perpetual source is not a complete pre-2024 one-minute clock")
    for name in ("open", "high", "low", "close"):
        futures[name] = pd.to_numeric(futures[name], errors="raise")
    _validate_ohlc(futures)

    spot = pd.read_csv(spot_path, compression="infer")
    spot["date"] = _dates(spot)
    if len(spot) and spot["date"].max() >= pd.Timestamp(SELECTION_END):
        raise RuntimeError("spot source opened 2024+ rows")
    repaired, spot_diagnostics = reconstruct_spot(
        futures, spot, cushion=cfg.incomplete_spot_cushion
    )
    market = futures.rename(
        columns={name: f"perp_{name}" for name in ("open", "high", "low", "close")}
    ).merge(repaired, on="date", validate="one_to_one")

    funding = pd.read_csv(funding_path, compression="infer")
    required_funding = {"date", "funding_rate", "mark_price"}
    if not required_funding.issubset(funding.columns):
        raise ValueError(
            f"funding source missing columns: {sorted(required_funding - set(funding.columns))}"
        )
    funding = funding[list(required_funding)].copy()
    funding["date"] = _dates(funding)
    funding["funding_rate"] = pd.to_numeric(funding["funding_rate"], errors="raise")
    funding["reported_mark_price"] = pd.to_numeric(funding["mark_price"], errors="coerce")
    funding = funding.drop(columns="mark_price").sort_values("date").reset_index(drop=True)
    if funding["date"].duplicated().any() or not funding["date"].is_monotonic_increasing:
        raise ValueError("funding timestamps must be unique and increasing")
    if len(funding) and funding["date"].max() >= pd.Timestamp(SELECTION_END):
        raise RuntimeError("funding source opened 2024+ rows")
    if not np.isfinite(funding["funding_rate"].to_numpy(float)).all():
        raise ValueError("funding rates must be finite")

    funding, funding_diagnostics = map_funding_to_market(market, funding)
    diagnostics = {
        **spot_diagnostics,
        **funding_diagnostics,
        "market_rows": int(len(market)),
        "funding_events": int(len(funding)),
        "oos_rows_opened": 0,
    }
    return Sources(market=market, funding=funding, source_hashes=hashes, diagnostics=diagnostics)


def policy_grid() -> list[Policy]:
    rows = []
    for lookback, entry, factor, min_hold in itertools.product(
        (3, 9, 21, 42),
        (0.0, 1e-5, 2e-5, 5e-5),
        (0.0, 0.5),
        (3, 9, 21),
    ):
        rows.append(
            Policy(
                lookback_events=int(lookback),
                entry_threshold=float(entry),
                exit_threshold=float(entry * factor),
                min_hold_events=int(min_hold),
            )
        )
    return rows


def gate_actions(
    funding: pd.DataFrame,
    policy: Policy,
    *,
    delay_events: int = 0,
    invert: bool = False,
) -> tuple[dict[int, bool], list[dict[str, Any]]]:
    if policy.lookback_events <= 0 or policy.min_hold_events <= 0 or delay_events < 0:
        raise ValueError("policy event counts must be positive and delay non-negative")
    rates = funding["funding_rate"].to_numpy(float)
    means = (
        pd.Series(rates)
        .rolling(policy.lookback_events, min_periods=policy.lookback_events)
        .mean()
        .shift(delay_events)
        .to_numpy(float)
    )
    active = False
    held_events = 0
    actions: dict[int, bool] = {}
    trace: list[dict[str, Any]] = []
    for position, row in enumerate(funding.itertuples(index=False)):
        if active:
            held_events += 1
        mean = float(means[position])
        if not math.isfinite(mean):
            continue
        if invert:
            enter = mean <= -policy.entry_threshold
            exit_now = mean >= -policy.exit_threshold
        else:
            enter = mean >= policy.entry_threshold
            exit_now = mean <= policy.exit_threshold
        target: bool | None = None
        if not active and enter:
            target = True
            active = True
            held_events = 0
        elif active and held_events >= policy.min_hold_events and exit_now:
            target = False
            active = False
            held_events = 0
        if target is None:
            continue
        execution = int(row.exec_index)
        if execution in actions and actions[execution] != target:
            raise RuntimeError("multiple contradictory funding decisions map to one bar")
        actions[execution] = target
        trace.append(
            {
                "funding_position": int(position),
                "funding_time": str(row.date),
                "execution_index": execution,
                "target_active": target,
                "trailing_mean": mean,
            }
        )
    return actions, trace


def schedule_hash(actions: dict[int, bool]) -> str:
    return _json_hash([[int(index), bool(target)] for index, target in sorted(actions.items())])


def _target_before(actions: dict[int, bool], index: int) -> bool:
    target = False
    for action_index, value in sorted(actions.items()):
        if action_index > index:
            break
        target = bool(value)
    return target


def _trade_cost(
    old_spot: float,
    old_perp: float,
    new_spot: float,
    new_perp: float,
    spot_price: float,
    perp_price: float,
    costs: CostModel,
) -> tuple[float, float]:
    spot_turnover = abs(new_spot - old_spot) * spot_price
    perp_turnover = abs(new_perp - old_perp) * perp_price
    return (
        spot_turnover * costs.spot_rate + perp_turnover * costs.perp_rate,
        spot_turnover + perp_turnover,
    )


def target_delta_neutral_quantities(
    equity: float,
    spot_price: float,
    perp_price: float,
    gross_exposure: float,
) -> tuple[float, float]:
    if min(equity, spot_price, perp_price, gross_exposure) <= 0.0:
        raise ValueError("delta-neutral target inputs must be positive")
    quantity = gross_exposure * equity / (spot_price + perp_price)
    return float(quantity), float(quantity)


def _daily_returns(equity: np.ndarray, dates: pd.Series, initial: float) -> pd.Series:
    series = pd.Series(equity, index=pd.DatetimeIndex(dates))
    closes = series.resample("1D").last().dropna()
    prior = pd.concat([pd.Series([initial], index=[closes.index[0] - pd.Timedelta("1ns")]), closes])
    return prior.pct_change().iloc[1:].replace([np.inf, -np.inf], np.nan).dropna()


def simulate_window(
    sources: Sources,
    actions: dict[int, bool],
    *,
    start: str,
    end: str,
    cfg: Config,
    costs: CostModel | None = None,
    include_funding: bool = True,
    force_initial_active: bool | None = None,
) -> dict[str, Any]:
    market = sources.market
    funding = sources.funding
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if not start_ts < end_ts <= pd.Timestamp(SELECTION_END):
        raise ValueError("simulation window must remain inside the sealed pre-2024 interval")
    dates = market["date"]
    left = int(np.searchsorted(dates.to_numpy(dtype="datetime64[ns]"), start_ts.to_datetime64()))
    right = int(np.searchsorted(dates.to_numpy(dtype="datetime64[ns]"), end_ts.to_datetime64()))
    if left >= right:
        raise ValueError("simulation window contains no market bars")
    costs = costs or CostModel(
        cfg.spot_fee_rate + cfg.spot_slippage_rate,
        cfg.perp_fee_rate + cfg.perp_slippage_rate,
    )
    if min(costs.spot_rate, costs.perp_rate) < 0.0:
        raise ValueError("execution cost rates cannot be negative")

    s_open = market["spot_open"].to_numpy(float)
    s_high = market["spot_high"].to_numpy(float)
    s_low = market["spot_low"].to_numpy(float)
    s_close = market["spot_close"].to_numpy(float)
    p_open = market["perp_open"].to_numpy(float)
    p_high = market["perp_high"].to_numpy(float)
    p_low = market["perp_low"].to_numpy(float)
    p_close = market["perp_close"].to_numpy(float)
    event_settlement = funding["settlement_index"].to_numpy(int)
    event_times = funding["date"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    event_rates = funding["funding_rate"].to_numpy(float)
    event_marks = funding["settlement_mark"].to_numpy(float)

    action_indices = [index for index in actions if left < index < right]
    funding_indices = sorted(
        set(int(value) for value in event_settlement if left <= value < right)
    )
    date_slice = dates.iloc[left:right]
    rebalance_indices = (
        np.flatnonzero(
            (date_slice.dt.hour.to_numpy() == 0)
            & (date_slice.dt.minute.to_numpy() == 5)
        )
        + left
    ).tolist()
    boundaries = sorted(set([left, right, *action_indices, *funding_indices, *rebalance_indices]))
    events_at: dict[int, list[int]] = {}
    for event_position, index in enumerate(event_settlement):
        if left <= int(index) < right:
            events_at.setdefault(int(index), []).append(event_position)

    close_equity = np.full(right - left, np.nan, dtype=float)
    favorable_equity = np.full(right - left, np.nan, dtype=float)
    adverse_equity = np.full(right - left, np.nan, dtype=float)
    equity = peak = 1.0
    strict_mdd = close_mdd = 0.0
    q_spot = q_perp = 0.0
    active = False
    entry_time_ns: int | None = None
    episodes = received_events = 0
    funding_cash = transaction_cost = turnover = 0.0
    active_days: set[str] = set()
    entry_times: list[str] = []
    exit_times: list[str] = []
    previous_close_index: int | None = None
    initial_target = (
        _target_before(actions, left)
        if force_initial_active is None
        else bool(force_initial_active)
    )

    for boundary_position, index in enumerate(boundaries[:-1]):
        next_index = boundaries[boundary_position + 1]
        timestamp_ns = int(pd.Timestamp(dates.iloc[index]).value)
        if previous_close_index is not None and active:
            equity += q_spot * (s_open[index] - s_close[previous_close_index])
            equity -= q_perp * (p_open[index] - p_close[previous_close_index])
            peak = max(peak, equity)
            strict_mdd = max(strict_mdd, 1.0 - equity / peak)

        # Funding settled at the event belongs only to a position opened strictly
        # before that event.  It is accounted before any decision at this bar.
        for event_position in events_at.get(index, []):
            if (
                active
                and entry_time_ns is not None
                and entry_time_ns < int(event_times[event_position])
            ):
                mark = float(event_marks[event_position])
                if not math.isfinite(mark) or mark <= 0.0:
                    raise RuntimeError("active funding event lacks a causal settlement mark")
                cash = q_perp * mark * float(event_rates[event_position]) if include_funding else 0.0
                equity += cash
                funding_cash += cash
                received_events += 1
                peak = max(peak, equity)
                strict_mdd = max(strict_mdd, 1.0 - equity / peak)

        desired = initial_target if index == left else actions.get(index, active)
        changed = bool(desired) != active
        rebalance = bool(active and dates.iloc[index].hour == 0 and dates.iloc[index].minute == 5)
        if changed or rebalance:
            if bool(desired):
                target_spot, target_perp = target_delta_neutral_quantities(
                    equity,
                    s_open[index],
                    p_open[index],
                    cfg.gross_exposure,
                )
            else:
                target_spot = target_perp = 0.0
            cost, traded = _trade_cost(
                q_spot,
                q_perp,
                target_spot,
                target_perp,
                s_open[index],
                p_open[index],
                costs,
            )
            equity -= cost
            transaction_cost += cost
            turnover += traded
            q_spot, q_perp = target_spot, target_perp
            if changed and bool(desired):
                active = True
                entry_time_ns = timestamp_ns
                episodes += 1
                entry_times.append(str(dates.iloc[index]))
            elif changed:
                active = False
                entry_time_ns = None
                exit_times.append(str(dates.iloc[index]))
            peak = max(peak, equity)
            strict_mdd = max(strict_mdd, 1.0 - equity / peak)

        segment = slice(index, next_index)
        local = slice(index - left, next_index - left)
        if active:
            base = equity
            favorable = (
                base
                + q_spot * (s_high[segment] - s_open[index])
                - q_perp * (p_low[segment] - p_open[index])
            )
            adverse = (
                base
                + q_spot * (s_low[segment] - s_open[index])
                - q_perp * (p_high[segment] - p_open[index])
            )
            closes = (
                base
                + q_spot * (s_close[segment] - s_open[index])
                - q_perp * (p_close[segment] - p_open[index])
            )
            # Strict path: the cross-venue extrema can be asynchronous inside the
            # minute, so first admit spot-high/perp-low into the global HWM and
            # then mark spot-low/perp-high as the adverse held-position extreme.
            favorable = np.maximum.reduce(
                [np.full(len(closes), base, dtype=float), favorable, closes]
            )
            adverse = np.minimum.reduce(
                [np.full(len(closes), base, dtype=float), adverse, closes]
            )
            active_days.update(str(value.date()) for value in dates.iloc[index:next_index])
        else:
            size = next_index - index
            favorable = adverse = closes = np.full(size, equity, dtype=float)
        running_peak = np.maximum.accumulate(np.concatenate(([peak], favorable)))[1:]
        strict_mdd = max(strict_mdd, float(np.max(1.0 - adverse / running_peak)))
        peak = max(peak, float(np.max(favorable)))
        favorable_equity[local] = favorable
        adverse_equity[local] = adverse
        close_equity[local] = closes
        equity = float(closes[-1])
        previous_close_index = next_index - 1

    # Crystallize both legs at the final executable close inside the split.
    if active and previous_close_index is not None:
        cost, traded = _trade_cost(
            q_spot,
            q_perp,
            0.0,
            0.0,
            s_close[previous_close_index],
            p_close[previous_close_index],
            costs,
        )
        equity -= cost
        transaction_cost += cost
        turnover += traded
        close_equity[-1] = equity
        adverse_equity[-1] = min(adverse_equity[-1], equity)
        strict_mdd = max(strict_mdd, 1.0 - equity / peak)
        exit_times.append(str(dates.iloc[previous_close_index] + pd.Timedelta("1min")))

    if not np.isfinite(close_equity).all() or not np.isfinite(adverse_equity).all():
        raise RuntimeError("simulation left non-finite equity marks")
    close_peaks = np.maximum.accumulate(np.concatenate(([1.0], close_equity)))[1:]
    close_mdd = float(np.max(1.0 - close_equity / close_peaks))
    years = (end_ts - start_ts).total_seconds() / (365.25 * 86_400.0)
    absolute_return = (equity - 1.0) * 100.0
    cagr = (equity ** (1.0 / years) - 1.0) * 100.0 if equity > 0.0 else -100.0
    strict_pct = min(max(strict_mdd * 100.0, 0.0), 100.0)
    daily = _daily_returns(close_equity, date_slice, 1.0)
    perp_daily = pd.Series(
        p_close[left:right], index=pd.DatetimeIndex(date_slice)
    ).resample("1D").last().pct_change().dropna()
    joined = pd.concat([daily.rename("strategy"), perp_daily.rename("btc")], axis=1).dropna()
    beta = (
        float(np.cov(joined["strategy"], joined["btc"], ddof=1)[0, 1] / np.var(joined["btc"], ddof=1))
        if len(joined) >= 3 and float(np.var(joined["btc"], ddof=1)) > 0.0
        else 0.0
    )
    stats = {
        "absolute_return_pct": float(absolute_return),
        "cagr_pct": float(cagr),
        "strict_mdd_pct": float(strict_pct),
        "close_mdd_pct": float(max(close_mdd, 0.0) * 100.0),
        "cagr_to_strict_mdd": float(cagr / strict_pct) if strict_pct > 1e-12 else 0.0,
        "calendar_years": float(years),
        "episodes": int(episodes),
        "active_days": int(len(active_days)),
        "funding_events_received": int(received_events),
        "funding_cash_pct_initial": float(funding_cash * 100.0),
        "transaction_cost_pct_initial": float(transaction_cost * 100.0),
        "gross_turnover_x_initial": float(turnover),
        "nonzero_daily_pnl_days": int((daily.abs() > 1e-12).sum()),
        "daily_btc_beta": beta,
        "entry_times": entry_times,
        "exit_times": exit_times,
    }
    return {"stats": stats, "daily_returns": daily, "equity": close_equity}


def _window_stats(
    sources: Sources,
    actions: dict[int, bool],
    cfg: Config,
    *,
    costs: CostModel | None = None,
    include_funding: bool = True,
    windows: Iterable[str] = WINDOWS,
    force_initial_active: bool | None = None,
) -> dict[str, dict[str, Any]]:
    return {
        name: simulate_window(
            sources,
            actions,
            start=WINDOWS[name][0],
            end=WINDOWS[name][1],
            cfg=cfg,
            costs=costs,
            include_funding=include_funding,
            force_initial_active=force_initial_active,
        )["stats"]
        for name in windows
    }


def _eligibility(stats: dict[str, dict[str, Any]]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for name in HALF_WINDOWS:
        row = stats[name]
        if row["absolute_return_pct"] <= 0.0:
            failures.append(f"{name}:nonpositive")
        if row["active_days"] < 20:
            failures.append(f"{name}:active_days<20")
        if row["funding_events_received"] < 20:
            failures.append(f"{name}:funding_events<20")
    for name, min_cagr in (("fit_2020_2022", 3.0), ("select_2023", 2.0)):
        if stats[name]["cagr_pct"] < min_cagr:
            failures.append(f"{name}:cagr<{min_cagr}")
        if stats[name]["cagr_to_strict_mdd"] < 3.0:
            failures.append(f"{name}:ratio<3")
    return not failures, failures


def _rank_key(row: dict[str, Any]) -> tuple[float, float, float]:
    stats = row["stats"]
    min_full_ratio = min(
        stats["fit_2020_2022"]["cagr_to_strict_mdd"],
        stats["select_2023"]["cagr_to_strict_mdd"],
    )
    min_half_return = min(stats[name]["absolute_return_pct"] for name in HALF_WINDOWS)
    total_return = (
        stats["fit_2020_2022"]["absolute_return_pct"]
        + stats["select_2023"]["absolute_return_pct"]
    )
    return float(min_full_ratio), float(min_half_return), float(total_return)


def block_bootstrap(daily: pd.Series, cfg: Config) -> dict[str, Any]:
    values = daily.to_numpy(float)
    if len(values) < 14:
        return {"days": int(len(values)), "valid": False}
    block = 7
    blocks = [values[i : i + block] for i in range(0, len(values), block)]
    rng = np.random.default_rng(cfg.bootstrap_seed)
    means = np.empty(cfg.bootstrap_samples, dtype=float)
    needed = len(blocks)
    for sample in range(cfg.bootstrap_samples):
        draw = np.concatenate([blocks[index] for index in rng.integers(0, needed, needed)])[: len(values)]
        means[sample] = float(draw.mean())
    standard_error = float(values.std(ddof=1) / math.sqrt(len(values)))
    return {
        "valid": True,
        "days": int(len(values)),
        "nonzero_days": int((np.abs(values) > 1e-12).sum()),
        "mean_daily_bps": float(values.mean() * 10_000.0),
        "naive_t_stat": float(values.mean() / standard_error) if standard_error > 0.0 else 0.0,
        "weekly_block_bootstrap_mean_daily_bps_ci95": [
            float(np.quantile(means, 0.025) * 10_000.0),
            float(np.quantile(means, 0.975) * 10_000.0),
        ],
        "weekly_block_bootstrap_p_mean_le_zero": float((means <= 0.0).mean()),
        "samples": int(cfg.bootstrap_samples),
        "block_days": block,
    }


def _policy_dict(policy: Policy) -> dict[str, Any]:
    return asdict(policy)


def _stats_table(stats: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "| 구간 | 절대수익 | CAGR | strict MDD | CAGR/MDD | 에피소드 | active days | funding events |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in stats.items():
        lines.append(
            f"| {name} | {row['absolute_return_pct']:.4f}% | {row['cagr_pct']:.4f}% | "
            f"{row['strict_mdd_pct']:.4f}% | {row['cagr_to_strict_mdd']:.4f} | "
            f"{row['episodes']} | {row['active_days']} | {row['funding_events_received']} |"
        )
    return lines


def write_docs(result: dict[str, Any], path: str) -> None:
    selected = result["selected"]
    lines = [
        "# Delta-neutral funding carry: pre-2024 frozen search",
        "",
        "## 결론",
        "",
        f"- 상태: **{result['decision']['status']}**",
        "- 2024년 이후 행은 열지 않았다(`oos_rows_opened=0`).",
        "- 구조: Binance BTCUSDT 현물 롱 + USD-M 무기한 숏의 BTC 수량을 정확히 동일하게 맞추고 합산 gross를 1배로 유지.",
        "- 정책 입력: 이미 정산된 funding-rate의 trailing mean만 사용하고 다음 5분봉 open에 집행.",
        "- DB 과거 현물/펀딩은 backfill된 비-PIT 스냅샷이므로 live forward proof 전에는 운영 승격 금지.",
        "- 분리 지갑에서 선물 담보가 고갈될 수 있으므로 통합마진/자동 담보이체 없이는 운영 승격 금지.",
        "",
        "## 선택 정책",
        "",
        "```json",
        json.dumps(selected["policy"], indent=2, sort_keys=True),
        "```",
        "",
        "## Pre-2024 성능",
        "",
        *_stats_table(selected["stats"]),
        "",
        "## 통제군",
        "",
    ]
    for name, stats in result["controls"].items():
        lines.extend([f"### {name}", "", *_stats_table(stats), ""])
    lines.extend(
        [
            "## 엄격성/누수 계약",
            "",
            "- 모든 창은 flat equity=1로 시작하며 과거 funding gate 상태만 전달한다.",
            "- funding event는 해당 시각보다 엄격히 뒤의 첫 5분봉 open에서만 gate를 바꾼다.",
            "- event 당시 이미 보유한 short만 funding을 받는다; 그 event로 진입한 포지션은 받지 않는다.",
            "- funding mark는 event 시각까지 완전히 끝난 마지막 선물 5분봉 close를 일관되게 사용한다.",
            "- strict MDD는 1분 내 비동시 basis dislocation까지 포함해 spot-high/perp-low HWM 뒤 spot-low/perp-high adverse를 적용한다.",
            "- 현물 누락/부분봉은 직전 완성 basis와 선물 OHLC로 복원하고 high/low를 고정 cushion만큼 확대한다.",
            "- 진입·청산·일일 리밸런싱 모두 두 leg의 실제 변경 notional에 fee+slippage를 부과한다.",
            "- CAGR 분모는 거래/보유일이 아니라 전체 달력 기간이다.",
            "",
            "## 직교성 판단",
            "",
            "이 단계에서는 방향성 알파와 다른 경제 메커니즘 및 일별 BTC beta를 확인한다. 기존의 entry/position Jaccard gate는 장시간 보유하는 market-neutral sleeve에 그대로 적용할 수 없으므로, frozen OOS 이후 동일 일별 손익의 Pearson 상관과 포트폴리오 한계기여를 주 판정으로 사용한다.",
            "",
            "## 소스 진단",
            "",
            "```json",
            json.dumps(result["sources"]["diagnostics"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def run(cfg: Config) -> dict[str, Any]:
    sources = load_sources(cfg)
    base_costs = CostModel(
        cfg.spot_fee_rate + cfg.spot_slippage_rate,
        cfg.perp_fee_rate + cfg.perp_slippage_rate,
    )
    raw_grid = policy_grid()
    unique: dict[str, dict[str, Any]] = {}
    for policy in raw_grid:
        actions, trace = gate_actions(sources.funding, policy)
        key = schedule_hash(actions)
        if key not in unique:
            unique[key] = {"policy": policy, "actions": actions, "trace": trace, "aliases": []}
        unique[key]["aliases"].append(_policy_dict(policy))

    searched: list[dict[str, Any]] = []
    for key, candidate in unique.items():
        stats = _window_stats(sources, candidate["actions"], cfg)
        eligible, failures = _eligibility(stats)
        searched.append(
            {
                "schedule_hash": key,
                "policy": _policy_dict(candidate["policy"]),
                "aliases": candidate["aliases"],
                "actions": int(len(candidate["actions"])),
                "stats": stats,
                "eligible": eligible,
                "eligibility_failures": failures,
            }
        )
    searched.sort(key=lambda row: (row["eligible"], *_rank_key(row)), reverse=True)
    eligible_rows = [row for row in searched if row["eligible"]]
    # Stress is a preregistered eligibility criterion, not a post-selection
    # report. Walk the already-fixed rank order and take the first candidate
    # that remains profitable under doubled execution costs.
    selection_pool = eligible_rows if eligible_rows else [searched[0]]
    selected = selection_pool[0]
    stress_stats: dict[str, dict[str, Any]] = {}
    stress_positive = False
    for candidate in selection_pool:
        candidate_policy = Policy(**candidate["policy"])
        candidate_actions, _ = gate_actions(sources.funding, candidate_policy)
        candidate_stress = _window_stats(
            sources,
            candidate_actions,
            cfg,
            costs=CostModel(base_costs.spot_rate * 2.0, base_costs.perp_rate * 2.0),
            windows=("fit_2020_2022", "select_2023"),
        )
        candidate_stress_positive = all(
            row["absolute_return_pct"] > 0.0 for row in candidate_stress.values()
        )
        if candidate_stress_positive or not eligible_rows:
            selected = candidate
            stress_stats = candidate_stress
            stress_positive = candidate_stress_positive
            break
    selected_policy = Policy(**selected["policy"])
    selected_actions, selected_trace = gate_actions(sources.funding, selected_policy)
    selected["double_cost_stats"] = stress_stats
    selected["double_cost_positive"] = stress_positive

    inverted_actions, _ = gate_actions(sources.funding, selected_policy, invert=True)
    delayed_one, _ = gate_actions(sources.funding, selected_policy, delay_events=1)
    delayed_three, _ = gate_actions(sources.funding, selected_policy, delay_events=3)
    control_windows = ("fit_2020_2022", "select_2023")
    controls = {
        "always_carry": _window_stats(
            sources,
            {},
            cfg,
            windows=control_windows,
            force_initial_active=True,
        ),
        "inverted_gate": _window_stats(
            sources, inverted_actions, cfg, windows=control_windows
        ),
        "decision_delayed_1_event": _window_stats(
            sources, delayed_one, cfg, windows=control_windows
        ),
        "decision_delayed_3_events": _window_stats(
            sources, delayed_three, cfg, windows=control_windows
        ),
        "basis_only_zero_funding": _window_stats(
            sources,
            selected_actions,
            cfg,
            windows=control_windows,
            include_funding=False,
        ),
        "double_execution_cost": stress_stats,
    }
    significance: dict[str, Any] = {}
    for name in control_windows:
        simulation = simulate_window(
            sources,
            selected_actions,
            start=WINDOWS[name][0],
            end=WINDOWS[name][1],
            cfg=cfg,
        )
        significance[name] = block_bootstrap(simulation["daily_returns"], cfg)

    promoted = bool(selected["eligible"] and stress_positive)
    result: dict[str, Any] = {
        "protocol": {
            "name": "delta-neutral BTC spot/perpetual realized-funding carry",
            "stage": "pre2024_bounded_search_and_freeze",
            "selection_end_exclusive": SELECTION_END,
            "oos_opened": False,
            "future_research_already_viewed_globally": True,
            "database_snapshot_is_point_in_time": False,
            "ranking": (
                "eligible first; maximize min(fit full ratio, 2023 ratio), then minimum "
                "half-year absolute return, then combined full-window absolute return"
            ),
            "pre2024_statistics_are_post_selection_descriptive_only": True,
            "independent_validation_required": "single frozen 2024+ OOS plus live forward parity",
        },
        "config": asdict(cfg),
        "sources": {"hashes": sources.source_hashes, "diagnostics": sources.diagnostics},
        "search": {
            "raw_grid_rows": len(raw_grid),
            "unique_schedule_rows": len(unique),
            "eligible_rows": len(eligible_rows),
            "all_rows": searched,
        },
        "selected": selected,
        "selected_trace": selected_trace,
        "controls": controls,
        "significance": significance,
        "decision": {
            "status": "freeze_for_one_shot_oos" if promoted else "reject_pre2024",
            "eligible": bool(selected["eligible"]),
            "double_cost_positive": stress_positive,
            "live_promotion_blocked": True,
            "blockers": [
                "2024+ one-shot OOS not yet opened",
                "historical DB spot/funding source is backfilled rather than PIT",
                "requires live forward parity proof",
                "requires unified margin or automated collateral transfer/liquidation guard",
            ],
        },
    }
    output_path = Path(cfg.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    frozen_manifest = {
        "protocol": result["protocol"],
        "decision": result["decision"],
        "policy": selected["policy"],
        "schedule_hash": selected["schedule_hash"],
        "source_hashes": sources.source_hashes,
        "source_diagnostics": sources.diagnostics,
        "execution_contract": {
            "gross_exposure": cfg.gross_exposure,
            "quantity_contract": "spot BTC quantity == absolute perp-short BTC quantity",
            "notional_contract": "spot notional + absolute perp notional == gross_exposure * equity at rebalance",
            "spot_cost_rate": base_costs.spot_rate,
            "perp_cost_rate": base_costs.perp_rate,
            "rebalance": "daily at 00:05 UTC after funding settlement accounting",
            "gate_availability": "next 5m open strictly after realized funding timestamp",
            "funding_eligibility": "position entry time strictly before funding event",
            "funding_mark": sources.diagnostics["funding_mark_policy"],
            "strict_mdd": (
                "global HWM admits one-minute spot-high/perp-low favorable cross-venue extreme, "
                "then spot-low/perp-high adverse cross-venue extreme, plus all costs/funding debits"
            ),
            "split_policy": "flat start with prior-only gate state; forced two-leg close inside end",
            "calendar_cagr": True,
        },
        "pre2024_stats": selected["stats"],
        "double_cost_stats": stress_stats,
        "manifest_hash_without_self": "",
    }
    frozen_manifest["manifest_hash_without_self"] = _json_hash(
        {**frozen_manifest, "manifest_hash_without_self": ""}
    )
    manifest_path = Path(cfg.policy_manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(frozen_manifest, indent=2, sort_keys=True) + "\n")
    if cfg.docs_output:
        write_docs(result, cfg.docs_output)
    return result


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--futures-csv", default=DEFAULT_FUTURES)
    parser.add_argument("--spot-csv", default=DEFAULT_SPOT)
    parser.add_argument("--funding-csv", default=DEFAULT_FUNDING)
    parser.add_argument("--source-manifest", default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--policy-manifest", default=DEFAULT_POLICY_MANIFEST)
    parser.add_argument("--docs-output", default=DEFAULT_DOCS)
    parser.add_argument("--cutoff", default=SELECTION_END)
    parser.add_argument("--gross-exposure", type=float, default=1.0)
    parser.add_argument("--incomplete-spot-cushion", type=float, default=0.0025)
    args = vars(parser.parse_args())
    return Config(**args)


def main() -> None:
    result = run(parse_args())
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "policy": result["selected"]["policy"],
                "stats": result["selected"]["stats"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
