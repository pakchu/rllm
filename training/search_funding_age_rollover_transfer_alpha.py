from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from training.search_positioning_disagreement_alpha import _attach_delayed_metrics, _future_extreme, _simulate_no_stop
from training.search_positioning_hgb_path_alpha import _read_before

MARKET = "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz"
METRICS = "data/binance_um_metrics_BTCUSDT_5m_2020-09-01_2026-06-01.csv.gz"
FUNDING = "data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz"
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


def prior_z(values: pd.Series, window: int = 2016) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    prior = values.shift(1)
    mean = prior.rolling(window, min_periods=max(288, window // 2)).mean()
    std = prior.rolling(window, min_periods=max(288, window // 2)).std(ddof=0).replace(0.0, np.nan)
    return (values - mean) / std


def load_pre2024() -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    market = _read_before(MARKET, "date", "2024-01-01")
    metrics = _read_before(METRICS, "create_time", "2024-01-01")
    funding = _read_before(FUNDING, "date", "2024-01-01")
    market = _attach_delayed_metrics(market, metrics, tolerance="5min", delay_bars=1)
    market["date"] = pd.to_datetime(market["date"], utc=True).dt.tz_convert(None)
    dates = pd.to_datetime(market["date"])
    event_rate = np.full(len(market), np.nan, dtype=float)
    exact = pd.to_datetime(pd.to_numeric(funding["funding_time"], errors="raise"), unit="ms", utc=True).dt.tz_convert(None)
    date_values = dates.to_numpy(dtype="datetime64[ns]")
    for t, rate in zip(exact, pd.to_numeric(funding["funding_rate"], errors="coerce"), strict=True):
        known = pd.Timestamp(t).ceil("5min")
        pos = int(np.searchsorted(date_values, np.datetime64(known), side="left"))
        if pos < len(event_rate) and np.isfinite(rate):
            event_rate[pos] = float(rate)
    if dates.max() >= pd.Timestamp("2024-01-01"):
        raise RuntimeError("future market opened")
    source = pd.to_datetime(market["positioning_source_time"], errors="coerce")
    if source.notna().any() and source.max() >= pd.Timestamp("2024-01-01"):
        raise RuntimeError("future metrics opened")
    return market, dates, event_rate


def owner_state(market: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    global_ratio = np.log(pd.to_numeric(market["count_long_short_ratio"], errors="coerce").where(lambda x: x > 0))
    quote = pd.to_numeric(market["quote_asset_volume"], errors="coerce")
    taker_buy = pd.to_numeric(market["taker_buy_quote"], errors="coerce")
    flow = ((2.0 * taker_buy / quote.replace(0.0, np.nan)) - 1.0).rolling(12, min_periods=12).mean()
    owner = np.tanh(0.5 * prior_z(global_ratio, 288) + 0.5 * prior_z(flow, 288))
    return owner, owner - owner.shift(12)


def build_features(
    market: pd.DataFrame,
    event_rate: np.ndarray,
    *,
    min_age_settlements: int,
    half_life_bars: int,
    fake_event_offset_bars: int = 0,
    carry_blind: bool = False,
) -> pd.DataFrame:
    if fake_event_offset_bars:
        shifted = np.full_like(event_rate, np.nan)
        shifted[fake_event_offset_bars:] = event_rate[:-fake_event_offset_bars]
        event_rate = shifted
    owner, owner_change = owner_state(market)
    owner_np = owner.to_numpy(float)
    owner_change_np = owner_change.to_numpy(float)
    oi = pd.to_numeric(market["sum_open_interest"], errors="coerce").to_numpy(float)
    log_price = np.log(pd.to_numeric(market["close"], errors="coerce").to_numpy(float))
    max_age = 24
    long_w = np.zeros(max_age + 1)
    short_w = np.zeros(max_age + 1)
    long_entry = np.zeros(max_age + 1)
    short_entry = np.zeros(max_age + 1)
    long_funding = np.zeros(max_age + 1)
    short_funding = np.zeros(max_age + 1)
    decay = float(np.exp(-np.log(2.0) / half_life_bars))
    long_burdens = np.full(len(market), np.nan)
    short_burdens = np.full(len(market), np.nan)
    signed_transfer = np.full(len(market), np.nan)
    eligible_weight = np.full(len(market), np.nan)

    def age_buckets() -> None:
        for arr in [long_w, short_w, long_entry, short_entry, long_funding, short_funding]:
            arr[-1] += arr[-2]
            arr[1:-1] = arr[:-2]
            arr[0] = 0.0

    for i in range(1, len(market)):
        if not (np.isfinite(log_price[i]) and np.isfinite(oi[i]) and oi[i] > 0 and np.isfinite(oi[i - 1]) and oi[i - 1] > 0):
            continue
        retention = decay * min(1.0, oi[i] / oi[i - 1])
        for arr in [long_w, short_w, long_entry, short_entry, long_funding, short_funding]:
            arr *= retention
        rate = event_rate[i]
        if np.isfinite(rate):
            age_buckets()
            if not carry_blind:
                long_funding += long_w * rate
                short_funding -= short_w * rate

        ages = slice(min_age_settlements, None)
        lw = long_w[ages]
        sw = short_w[ages]
        lb = np.divide(long_entry[ages], lw, out=np.full_like(lw, np.nan), where=lw > 1e-15)
        sb = np.divide(short_entry[ages], sw, out=np.full_like(sw, np.nan), where=sw > 1e-15)
        lf = np.divide(long_funding[ages], lw, out=np.full_like(lw, np.nan), where=lw > 1e-15)
        sf = np.divide(short_funding[ages], sw, out=np.full_like(sw, np.nan), where=sw > 1e-15)
        long_per_weight = np.maximum(lf - (log_price[i] - lb), 0.0)
        short_per_weight = np.maximum(sf + (log_price[i] - sb), 0.0)
        long_burden = float(np.nansum(lw * long_per_weight))
        short_burden = float(np.nansum(sw * short_per_weight))
        long_burdens[i] = long_burden
        short_burdens[i] = short_burden
        eligible_weight[i] = float(np.nansum(lw) + np.nansum(sw))
        if np.isfinite(owner_change_np[i]):
            long_transfer = long_burden * max(-owner_change_np[i], 0.0)
            short_transfer = short_burden * max(owner_change_np[i], 0.0)
            # Positive means burdened shorts are handing debt away -> long;
            # negative means burdened longs are handing debt away -> short.
            signed_transfer[i] = short_transfer - long_transfer

        delta = max(oi[i] - oi[i - 1], 0.0) / oi[i - 1]
        if delta > 0 and np.isfinite(owner_np[i]):
            long_add = delta * max(owner_np[i], 0.0)
            short_add = delta * max(-owner_np[i], 0.0)
            long_w[0] += long_add
            short_w[0] += short_add
            long_entry[0] += long_add * log_price[i]
            short_entry[0] += short_add * log_price[i]

    transfer_series = pd.Series(signed_transfer)
    intensity_z = prior_z(transfer_series.abs(), 2016)
    score = np.sign(transfer_series) * intensity_z.clip(lower=0.0)
    return pd.DataFrame(
        {
            "long_burden": long_burdens,
            "short_burden": short_burdens,
            "eligible_weight": eligible_weight,
            "signed_transfer": transfer_series,
            "intensity_z": intensity_z,
            "score": score,
            "owner": owner,
            "owner_change": owner_change,
        }
    ).replace([np.inf, -np.inf], np.nan)


def fit_abs_threshold(score: pd.Series, dates: pd.Series, q: float = 0.9) -> float:
    a, b = WINDOWS["fit"]
    mask = (dates >= pd.Timestamp(a)) & (dates < pd.Timestamp(b))
    values = score.loc[mask].abs().dropna()
    if len(values) < 10_000:
        raise ValueError(f"insufficient fit scores {len(values)}")
    return float(values.quantile(q))


def signals(features: pd.DataFrame, threshold: float, flip: bool = False) -> tuple[np.ndarray, np.ndarray]:
    score = features["score"].to_numpy(float)
    active = np.isfinite(score) & (np.abs(score) >= threshold)
    onset = active & ~np.r_[False, active[:-1]]
    side = np.sign(score)
    if flip:
        side = -side
    return onset & (side > 0), onset & (side < 0)


def simulate(market: pd.DataFrame, dates: pd.Series, la: np.ndarray, sa: np.ndarray, hold: int, extremes: tuple[np.ndarray, np.ndarray]) -> dict:
    return {w: _simulate_no_stop(market, dates, la, sa, window=w, hold_bars=hold, stride_bars=1, leverage=.5, fee_rate=.0005, slippage_rate=.0001, extremes=extremes, windows=WINDOWS) for w in WINDOWS}


def rank_key(stats: dict) -> tuple:
    enough = stats["fit"]["trades"] >= 80 and stats["select_2023"]["trades"] >= 24 and min(stats[w]["trades"] for w in ["select_2023_h1", "select_2023_h2"]) >= 8
    positive = sum(stats[w]["return_pct"] > 0 for w in SEGMENTS)
    core = [stats[w]["ratio"] for w in ["fit", "select_2023", "select_2023_h1", "select_2023_h2"]]
    return enough, positive, min(core), float(np.median(core)), stats["select_2023"]["trades"]


def print_stats(title: str, stats: dict) -> None:
    print("\n" + title)
    for w in ["fit", "select_2023", *SEGMENTS]:
        s = stats[w]
        print(w, f"ret={s['return_pct']:.2f}", f"cagr={s['cagr_pct']:.2f}", f"mdd={s['strict_mdd_pct']:.2f}", f"ratio={s['ratio']:.2f}", f"n={s['trades']}", f"L/S={s['longs']}/{s['shorts']}")


def main() -> None:
    market, dates, event_rate = load_pre2024()
    holds = [72, 144]
    extremes = {h: (_future_extreme(market.low.to_numpy(float), h, "min"), _future_extreme(market.high.to_numpy(float), h, "max")) for h in holds}
    rows = []
    banks = {}
    for n_age, half_life in itertools.product([1, 3, 6], [288, 864]):
        f = build_features(market, event_rate, min_age_settlements=n_age, half_life_bars=half_life)
        banks[(n_age, half_life, "primary")] = f
        threshold = fit_abs_threshold(f["score"], dates)
        for hold in holds:
            la, sa = signals(f, threshold)
            stats = simulate(market, dates, la, sa, hold, extremes[hold])
            rows.append({"min_age": n_age, "half_life": half_life, "q": .9, "hold": hold, "threshold": threshold, "rank": rank_key(stats), "stats": stats})
    rows.sort(key=lambda r: r["rank"], reverse=True)
    print("candidates", len(rows), "funding_events", int(np.isfinite(event_rate).sum()))
    for i, r in enumerate(rows, 1):
        print_stats(f"RANK {i} age{r['min_age']} hl{r['half_life']} hold{r['hold']} rank={r['rank']}", r["stats"])
    top = rows[0]
    controls = {}
    variants = {
        "direction_flip": dict(min_age_settlements=top["min_age"], half_life_bars=top["half_life"]),
        "fake_settlement_plus_4h": dict(min_age_settlements=top["min_age"], half_life_bars=top["half_life"], fake_event_offset_bars=48),
        "age_blind": dict(min_age_settlements=0, half_life_bars=top["half_life"]),
        "carry_blind": dict(min_age_settlements=top["min_age"], half_life_bars=top["half_life"], carry_blind=True),
    }
    for name, kwargs in variants.items():
        f = banks.get((top["min_age"], top["half_life"], "primary")) if name == "direction_flip" else build_features(market, event_rate, **kwargs)
        th = fit_abs_threshold(f["score"], dates)
        la, sa = signals(f, th, flip=(name == "direction_flip"))
        controls[name] = simulate(market, dates, la, sa, top["hold"], extremes[top["hold"]])
        print_stats("CONTROL " + name, controls[name])
    Path("results/funding_age_rollover_transfer_alpha_scan_2026-07-13.json").write_text(json.dumps({"rows": [{**r, "rank": list(r["rank"])} for r in rows], "controls": controls}, indent=2))


if __name__ == "__main__":
    main()
