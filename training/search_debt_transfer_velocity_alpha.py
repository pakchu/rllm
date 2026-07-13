from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from training.search_positioning_disagreement_alpha import _attach_delayed_metrics, _future_extreme, _simulate_no_stop
from training.search_positioning_hgb_path_alpha import _read_before

MARKET = "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz"
SPOT = "data/cache_spot_premium_5m_2020-01-01_2026-06-01.csv.gz"
METRICS = "data/binance_um_metrics_BTCUSDT_5m_2020-09-01_2026-06-01.csv.gz"
WINDOWS = {
    "fit": ("2020-10-15", "2023-01-01"),
    "fit_2021_h1": ("2021-01-01", "2021-07-01"),
    "fit_2021_h2": ("2021-07-01", "2022-01-01"),
    "fit_2022_h1": ("2022-01-01", "2022-07-01"),
    "fit_2022_h2": ("2022-07-01", "2023-01-01"),
    "select_2023": ("2023-01-01", "2024-01-01"),
    "select_2023_h1": ("2023-01-01", "2023-07-01"),
    "select_2023_h2": ("2023-07-01", "2024-01-01"),
}
SEGMENTS = ["fit_2021_h1", "fit_2021_h2", "fit_2022_h1", "fit_2022_h2", "select_2023_h1", "select_2023_h2"]


def prior_z(values: pd.Series, window: int, minimum: int | None = None) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    prior = values.shift(1)
    minimum = max(288, window // 2) if minimum is None else minimum
    mean = prior.rolling(window, min_periods=minimum).mean()
    std = prior.rolling(window, min_periods=minimum).std(ddof=0).replace(0.0, np.nan)
    return (values - mean) / std


def load_pre2024() -> tuple[pd.DataFrame, pd.Series, dict]:
    market = _read_before(MARKET, "date", "2024-01-01")
    spot = _read_before(SPOT, "date", "2024-01-01")
    metrics = _read_before(METRICS, "create_time", "2024-01-01")
    market["date"] = pd.to_datetime(market["date"], utc=True).dt.tz_convert(None)
    spot["date"] = pd.to_datetime(spot["date"], utc=True).dt.tz_convert(None)
    market = market.sort_values("date").drop_duplicates("date", keep="last")
    spot = spot.sort_values("date").drop_duplicates("date", keep="last")
    market = market.merge(
        spot[["date", "spot_close", "spot_volume", "spot_rows"]],
        on="date",
        how="left",
        validate="one_to_one",
    ).reset_index(drop=True)
    market = _attach_delayed_metrics(market, metrics, tolerance="5min", delay_bars=1)
    dates = pd.to_datetime(market["date"])
    if dates.max() >= pd.Timestamp("2024-01-01"):
        raise RuntimeError("future market rows opened")
    source = pd.to_datetime(market["positioning_source_time"], errors="coerce")
    if source.notna().any() and source.max() >= pd.Timestamp("2024-01-01"):
        raise RuntimeError("future metrics rows opened")
    coverage = {}
    for year in [2021, 2022, 2023]:
        mask = (dates >= pd.Timestamp(f"{year}-01-01")) & (dates < pd.Timestamp(f"{year + 1}-01-01"))
        coverage[str(year)] = {
            "global_ratio": float(pd.to_numeric(market.loc[mask, "count_long_short_ratio"], errors="coerce").gt(0).mean()),
            "oi": float(pd.to_numeric(market.loc[mask, "sum_open_interest"], errors="coerce").gt(0).mean()),
            "oi_value": float(pd.to_numeric(market.loc[mask, "sum_open_interest_value"], errors="coerce").gt(0).mean()),
            "complete_spot": float(pd.to_numeric(market.loc[mask, "spot_rows"], errors="coerce").eq(5).mean()),
        }
    return market, dates, coverage


def build_features(market: pd.DataFrame, memory: int, acceptance_horizon: int, variant: str = "primary") -> pd.DataFrame:
    close = pd.to_numeric(market["close"], errors="coerce")
    log_price = np.log(close.where(close > 0))
    oi = np.log(pd.to_numeric(market["sum_open_interest"], errors="coerce").where(lambda x: x > 0))
    oi_value = np.log(pd.to_numeric(market["sum_open_interest_value"], errors="coerce").where(lambda x: x > 0))
    global_ratio = np.log(pd.to_numeric(market["count_long_short_ratio"], errors="coerce").where(lambda x: x > 0))
    quote = pd.to_numeric(market["quote_asset_volume"], errors="coerce")
    taker_buy = pd.to_numeric(market["taker_buy_quote"], errors="coerce")
    taker_flow = ((2.0 * taker_buy / quote.replace(0.0, np.nan)) - 1.0).rolling(12, min_periods=12).mean()
    owner = np.tanh(0.5 * prior_z(global_ratio, 288) + 0.5 * prior_z(taker_flow, 288))
    owner_change = owner - owner.shift(12)
    new_debt = (oi - oi.shift(1)).clip(lower=0.0)
    if variant == "no_transfer":
        impulse = new_debt * owner
    elif variant == "stale_owner":
        impulse = new_debt * owner_change.shift(288)
    else:
        impulse = new_debt * owner_change
    transfer_velocity = impulse.ewm(halflife=memory, min_periods=memory, adjust=False).mean()

    spot_close = pd.to_numeric(market["spot_close"], errors="coerce")
    spot_volume = pd.to_numeric(market["spot_volume"], errors="coerce")
    complete_spot = pd.to_numeric(market["spot_rows"], errors="coerce").eq(5)
    spot_notional_24h = (spot_close * spot_volume).where(complete_spot).rolling(288, min_periods=276).sum()
    spot_liquidity_growth = np.log(spot_notional_24h / spot_notional_24h.shift(288))
    debt_growth = oi_value - oi_value.shift(288)
    cash_gap = prior_z(debt_growth - spot_liquidity_growth, 2016)
    acceptance = np.sign(transfer_velocity) * (log_price - log_price.shift(acceptance_horizon))
    acceptance_z = prior_z(acceptance, 288)
    transfer_intensity_z = prior_z(transfer_velocity.abs(), 2016)
    if variant == "no_cash":
        cash_term = pd.Series(1.0, index=market.index)
    else:
        cash_term = cash_gap.clip(lower=0.0)
    score = transfer_intensity_z.clip(lower=0.0) * cash_term * acceptance_z.clip(lower=0.0)
    source_valid = (
        pd.to_numeric(market["sum_open_interest"], errors="coerce").gt(0)
        & pd.to_numeric(market["sum_open_interest_value"], errors="coerce").gt(0)
        & pd.to_numeric(market["count_long_short_ratio"], errors="coerce").gt(0)
        & pd.to_datetime(market["positioning_source_time"], errors="coerce").notna()
        & complete_spot
    )
    score = score.where(source_valid)
    return pd.DataFrame(
        {
            "owner": owner,
            "owner_change": owner_change,
            "new_debt": new_debt,
            "transfer_velocity": transfer_velocity,
            "transfer_intensity_z": transfer_intensity_z,
            "cash_gap": cash_gap,
            "acceptance_z": acceptance_z,
            "score": score,
        }
    )


def fit_threshold(score: pd.Series, dates: pd.Series, quantile: float) -> float:
    a, b = WINDOWS["fit"]
    mask = (dates >= pd.Timestamp(a)) & (dates < pd.Timestamp(b))
    values = pd.to_numeric(score.loc[mask], errors="coerce").dropna()
    values = values[values > 0]
    if len(values) < 1_000:
        raise ValueError(f"insufficient positive fit scores: {len(values)}")
    return float(values.quantile(quantile))


def signals(features: pd.DataFrame, threshold: float, flip: bool = False) -> tuple[np.ndarray, np.ndarray]:
    score = features["score"].to_numpy(float)
    velocity = features["transfer_velocity"].to_numpy(float)
    active = np.isfinite(score) & np.isfinite(velocity) & (score >= threshold)
    onset = active & ~np.r_[False, active[:-1]]
    side = -np.sign(velocity)
    if flip:
        side = -side
    return onset & (side > 0), onset & (side < 0)


def simulate(market: pd.DataFrame, dates: pd.Series, long_signal: np.ndarray, short_signal: np.ndarray, hold: int, extremes: tuple[np.ndarray, np.ndarray]) -> dict:
    return {
        name: _simulate_no_stop(
            market,
            dates,
            long_signal,
            short_signal,
            window=name,
            hold_bars=hold,
            stride_bars=1,
            leverage=0.5,
            fee_rate=0.0005,
            slippage_rate=0.0001,
            extremes=extremes,
            windows=WINDOWS,
        )
        for name in WINDOWS
    }


def rank_key(stats: dict) -> tuple:
    core = [stats["fit"], stats["select_2023"], stats["select_2023_h1"], stats["select_2023_h2"]]
    enough = stats["fit"]["trades"] >= 80 and stats["select_2023"]["trades"] >= 24 and min(stats[x]["trades"] for x in ["select_2023_h1", "select_2023_h2"]) >= 8
    positive = sum(stats[x]["return_pct"] > 0 for x in SEGMENTS)
    return enough, positive, min(x["ratio"] for x in core), float(np.median([x["ratio"] for x in core])), stats["select_2023"]["trades"]


def print_stats(title: str, stats: dict) -> None:
    print("\n" + title)
    for w in ["fit", "select_2023", *SEGMENTS]:
        s = stats[w]
        print(w, f"ret={s['return_pct']:.2f}", f"cagr={s['cagr_pct']:.2f}", f"mdd={s['strict_mdd_pct']:.2f}", f"ratio={s['ratio']:.2f}", f"n={s['trades']}", f"L/S={s['longs']}/{s['shorts']}")


def main() -> None:
    market, dates, coverage = load_pre2024()
    holds = [72, 144, 288]
    extremes = {h: (_future_extreme(market.low.to_numpy(float), h, "min"), _future_extreme(market.high.to_numpy(float), h, "max")) for h in holds}
    banks = {}
    rows = []
    for memory, horizon in itertools.product([72, 288], [72, 288]):
        features = build_features(market, memory, horizon)
        banks[(memory, horizon, "primary")] = features
        for q, hold in itertools.product([0.90, 0.95], holds):
            threshold = fit_threshold(features["score"], dates, q)
            long_signal, short_signal = signals(features, threshold)
            stats = simulate(market, dates, long_signal, short_signal, hold, extremes[hold])
            rows.append({"memory": memory, "horizon": horizon, "q": q, "hold": hold, "threshold": threshold, "rank": rank_key(stats), "stats": stats})
    rows.sort(key=lambda r: r["rank"], reverse=True)
    print("coverage", json.dumps(coverage, indent=2), "candidates", len(rows))
    for i, row in enumerate(rows[:10], 1):
        print_stats(f"RANK {i} {row['memory']} {row['horizon']} q{row['q']} hold{row['hold']} rank={row['rank']}", row["stats"])
    top = rows[0]
    controls = {}
    for variant in ["direction_flip", "no_transfer", "no_cash", "stale_owner"]:
        base_variant = "primary" if variant == "direction_flip" else variant
        features = banks.get((top["memory"], top["horizon"], base_variant))
        if features is None:
            features = build_features(market, top["memory"], top["horizon"], base_variant)
            banks[(top["memory"], top["horizon"], base_variant)] = features
        threshold = fit_threshold(features["score"], dates, top["q"])
        long_signal, short_signal = signals(features, threshold, flip=(variant == "direction_flip"))
        controls[variant] = simulate(market, dates, long_signal, short_signal, top["hold"], extremes[top["hold"]])
        print_stats("CONTROL " + variant, controls[variant])
    Path("results/debt_transfer_velocity_alpha_scan_2026-07-13.json").write_text(json.dumps({"coverage": coverage, "rows": [{**r, "rank": list(r["rank"])} for r in rows], "controls": controls}, indent=2))


if __name__ == "__main__":
    main()
