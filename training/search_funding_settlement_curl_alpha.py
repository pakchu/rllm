from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from training.search_funding_premium_external_state_gate_alpha import (
    ExternalStateGateConfig,
    _load_bundle,
)
from training.search_positioning_disagreement_alpha import _future_extreme, _simulate_no_stop


CFG = ExternalStateGateConfig(
    input_csv="data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz",
    metrics_csv="data/binance_um_metrics_BTCUSDT_5m_2020-09-01_2026-06-01.csv.gz",
    dvol_csv="data/deribit_btc_dvol_1h_2020-09-01_2026-06-02.csv.gz",
    funding_csv="data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz",
    premium_csv="data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz",
    output="/tmp/x",
    manifest_output="/tmp/y",
)

WINDOWS = {
    "fit": ("2020-12-01", "2023-01-01"),
    "fit_2021_h1": ("2021-01-01", "2021-07-01"),
    "fit_2021_h2": ("2021-07-01", "2022-01-01"),
    "fit_2022_h1": ("2022-01-01", "2022-07-01"),
    "fit_2022_h2": ("2022-07-01", "2023-01-01"),
    "select_2023": ("2023-01-01", "2024-01-01"),
    "select_2023_h1": ("2023-01-01", "2023-07-01"),
    "select_2023_h2": ("2023-07-01", "2024-01-01"),
}
SEGMENTS = [
    "fit_2021_h1",
    "fit_2021_h2",
    "fit_2022_h1",
    "fit_2022_h2",
    "select_2023_h1",
    "select_2023_h2",
]


def prior_event_z(values: pd.Series, window: int = 180, min_periods: int = 60) -> pd.Series:
    prior = values.shift(1)
    mean = prior.rolling(window, min_periods=min_periods).mean()
    std = prior.rolling(window, min_periods=min_periods).std(ddof=0).replace(0.0, np.nan)
    return ((values - mean) / std).clip(-8.0, 8.0)


def load_funding_events(cutoff: str) -> pd.DataFrame:
    frame = pd.read_csv(CFG.funding_csv, compression="infer")
    exact = pd.to_datetime(pd.to_numeric(frame["funding_time"], errors="raise"), unit="ms", utc=True).dt.tz_convert(None)
    frame = frame.assign(event_time=exact)
    frame = frame.loc[frame["event_time"] < pd.Timestamp(cutoff), ["event_time", "funding_rate"]]
    return frame.sort_values("event_time").drop_duplicates("event_time", keep="last").reset_index(drop=True)


def build_event_features(
    market: pd.DataFrame,
    dates: pd.Series,
    funding_events: pd.DataFrame,
    *,
    pre_bars: int,
    post_bars: int,
    event_offset_bars: int = 0,
) -> pd.DataFrame:
    premium = pd.to_numeric(market["premium_index"], errors="coerce").to_numpy(float)
    oi = np.log(pd.to_numeric(market["sum_open_interest"], errors="coerce").where(lambda x: x > 0.0)).to_numpy(float)
    date_values = dates.to_numpy(dtype="datetime64[ns]")

    rows: list[dict[str, float | int | pd.Timestamp]] = []
    for event in funding_events.itertuples(index=False):
        # Ceil to the first 5m bar that cannot precede the exact settlement;
        # the strategy waits a further post_bars and then executes next-bar open.
        known_bar_time = pd.Timestamp(event.event_time).ceil("5min") + pd.Timedelta(minutes=5 * event_offset_bars)
        event_pos = int(np.searchsorted(date_values, np.datetime64(known_bar_time), side="left"))
        signal_pos = event_pos + post_bars
        pre_pos = event_pos - pre_bars
        if pre_pos < 0 or signal_pos >= len(market):
            continue
        values = premium[[pre_pos, event_pos, signal_pos]].tolist() + oi[[pre_pos, event_pos, signal_pos]].tolist()
        if not np.isfinite(values).all():
            continue
        rows.append(
            {
                "event_time": pd.Timestamp(event.event_time),
                "known_bar_time": known_bar_time,
                "signal_time": pd.Timestamp(dates.iloc[signal_pos]),
                "signal_pos": signal_pos,
                "funding_rate": float(event.funding_rate),
                "funding_sign": float(np.sign(event.funding_rate)),
                "basis_pre": float(premium[event_pos] - premium[pre_pos]),
                "oi_pre": float(oi[event_pos] - oi[pre_pos]),
                "basis_post": float(premium[signal_pos] - premium[event_pos]),
                "oi_post": float(oi[signal_pos] - oi[event_pos]),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    for name in ("basis_pre", "oi_pre", "basis_post", "oi_post"):
        out[f"z_{name}"] = prior_event_z(out[name])
    out["curl"] = out["z_basis_pre"] * out["z_oi_post"] - out["z_oi_pre"] * out["z_basis_post"]
    out["trap"] = out["funding_sign"] * out["curl"]
    out["symmetric_dot"] = out["funding_sign"] * (
        out["z_basis_pre"] * out["z_oi_post"] + out["z_oi_pre"] * out["z_basis_post"]
    )
    out["aligned_basis_pre"] = out["funding_sign"] * out["z_basis_pre"]
    out["aligned_basis_post"] = out["funding_sign"] * out["z_basis_post"]
    return out.replace([np.inf, -np.inf], np.nan)


def fit_quantile(events: pd.DataFrame, column: str, q: float) -> float:
    fit = events.loc[
        (events["signal_time"] >= pd.Timestamp(WINDOWS["fit"][0]))
        & (events["signal_time"] < pd.Timestamp(WINDOWS["fit"][1])),
        column,
    ]
    fit = pd.to_numeric(fit, errors="coerce").dropna()
    if len(fit) < 100:
        raise ValueError(f"insufficient fit events for {column}: {len(fit)}")
    return float(fit.quantile(q))


def signals(
    events: pd.DataFrame,
    length: int,
    *,
    funding_q: float,
    score_q: float,
    mode: str,
    flip: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    abs_funding = events["funding_rate"].abs()
    funding_threshold = fit_quantile(events.assign(abs_funding=abs_funding), "abs_funding", funding_q)
    score_column = "trap" if mode in {"curl", "trapped"} else "symmetric_dot"
    score_threshold = fit_quantile(events, score_column, score_q)
    active = abs_funding.ge(funding_threshold) & events[score_column].ge(score_threshold)
    thresholds: dict[str, float] = {"abs_funding": funding_threshold, score_column: score_threshold}
    if mode == "trapped":
        pre_basis_threshold = fit_quantile(events, "aligned_basis_pre", 0.55)
        pre_oi_threshold = fit_quantile(events, "z_oi_pre", 0.55)
        post_basis_threshold = fit_quantile(events, "aligned_basis_post", 0.45)
        post_oi_threshold = fit_quantile(events, "z_oi_post", 0.25)
        active &= (
            events["aligned_basis_pre"].ge(pre_basis_threshold)
            & events["z_oi_pre"].ge(pre_oi_threshold)
            & events["aligned_basis_post"].le(post_basis_threshold)
            & events["z_oi_post"].ge(post_oi_threshold)
        )
        thresholds.update(
            {
                "aligned_basis_pre": pre_basis_threshold,
                "z_oi_pre": pre_oi_threshold,
                "aligned_basis_post": post_basis_threshold,
                "z_oi_post": post_oi_threshold,
            }
        )

    long_active = np.zeros(length, dtype=bool)
    short_active = np.zeros(length, dtype=bool)
    selected = events.loc[active & events["funding_sign"].ne(0.0)]
    sides = -selected["funding_sign"].to_numpy(float)
    if flip:
        sides = -sides
    positions = selected["signal_pos"].to_numpy(int)
    long_active[positions[sides > 0.0]] = True
    short_active[positions[sides < 0.0]] = True
    return long_active, short_active, thresholds


def funding_only_signals(events: pd.DataFrame, length: int, funding_q: float, flip: bool = False) -> tuple[np.ndarray, np.ndarray]:
    abs_funding = events["funding_rate"].abs()
    threshold = fit_quantile(events.assign(abs_funding=abs_funding), "abs_funding", funding_q)
    selected = events.loc[abs_funding.ge(threshold) & events["funding_sign"].ne(0.0)]
    sides = -selected["funding_sign"].to_numpy(float)
    if flip:
        sides = -sides
    positions = selected["signal_pos"].to_numpy(int)
    long_active = np.zeros(length, dtype=bool)
    short_active = np.zeros(length, dtype=bool)
    long_active[positions[sides > 0.0]] = True
    short_active[positions[sides < 0.0]] = True
    return long_active, short_active


def simulate_all(market: pd.DataFrame, dates: pd.Series, long_active: np.ndarray, short_active: np.ndarray, hold_bars: int, extremes: tuple[np.ndarray, np.ndarray]) -> dict[str, dict]:
    return {
        window: _simulate_no_stop(
            market,
            dates,
            long_active,
            short_active,
            window=window,
            hold_bars=hold_bars,
            stride_bars=1,
            leverage=0.5,
            fee_rate=0.0005,
            slippage_rate=0.0001,
            extremes=extremes,
            windows=WINDOWS,
        )
        for window in WINDOWS
    }


def score_row(stats: dict[str, dict]) -> tuple:
    core = [stats["fit"], stats["select_2023"], stats["select_2023_h1"], stats["select_2023_h2"]]
    positive_segments = sum(stats[name]["return_pct"] > 0.0 for name in SEGMENTS)
    min_core_ratio = min(item["ratio"] for item in core)
    median_core_ratio = float(np.median([item["ratio"] for item in core]))
    trade_ok = stats["fit"]["trades"] >= 80 and stats["select_2023"]["trades"] >= 24
    halves_ok = stats["select_2023_h1"]["trades"] >= 8 and stats["select_2023_h2"]["trades"] >= 8
    return (trade_ok and halves_ok, positive_segments, min_core_ratio, median_core_ratio, stats["select_2023"]["trades"])


def print_stats(name: str, stats: dict[str, dict]) -> None:
    print(f"\n{name}")
    for window in ["fit", "select_2023", *SEGMENTS]:
        s = stats[window]
        print(
            window,
            f"ret={s['return_pct']:.2f}",
            f"cagr={s['cagr_pct']:.2f}",
            f"mdd={s['strict_mdd_pct']:.2f}",
            f"ratio={s['ratio']:.2f}",
            f"n={s['trades']}",
            f"L/S={s['longs']}/{s['shorts']}",
        )


def main() -> None:
    market, dates, *_ = _load_bundle(CFG, cutoff="2024-01-01")
    raw_events = load_funding_events("2024-01-01")
    holds = [24, 48, 96]
    extremes = {
        hold: (
            _future_extreme(market["low"].to_numpy(float), hold, "min"),
            _future_extreme(market["high"].to_numpy(float), hold, "max"),
        )
        for hold in holds
    }
    banks: dict[tuple[int, int, int], pd.DataFrame] = {}
    rows: list[dict] = []
    # Premium index closes hourly, so a sub-hour post window contains no new
    # basis observation and cannot identify response ordering.
    for pre_bars, post_bars in itertools.product([12, 24], [12, 24]):
        events = build_event_features(
            market,
            dates,
            raw_events,
            pre_bars=pre_bars,
            post_bars=post_bars,
        )
        banks[(pre_bars, post_bars, 0)] = events
        for funding_q, score_q, mode, hold_bars in itertools.product(
            [0.70, 0.85], [0.75, 0.85, 0.90], ["curl", "trapped"], holds
        ):
            long_active, short_active, thresholds = signals(
                events,
                len(market),
                funding_q=funding_q,
                score_q=score_q,
                mode=mode,
            )
            stats = simulate_all(market, dates, long_active, short_active, hold_bars, extremes[hold_bars])
            rows.append(
                {
                    "pre_bars": pre_bars,
                    "post_bars": post_bars,
                    "funding_q": funding_q,
                    "score_q": score_q,
                    "mode": mode,
                    "hold_bars": hold_bars,
                    "thresholds": thresholds,
                    "score": score_row(stats),
                    "stats": stats,
                }
            )
    rows.sort(key=lambda row: row["score"], reverse=True)
    print(f"loaded market={len(market)} events={len(raw_events)} candidates={len(rows)}")
    for rank, row in enumerate(rows[:12], start=1):
        print_stats(
            f"RANK {rank}: pre={row['pre_bars']} post={row['post_bars']} fq={row['funding_q']} sq={row['score_q']} mode={row['mode']} hold={row['hold_bars']} score={row['score']}",
            row["stats"],
        )

    top = rows[0]
    top_events = banks[(top["pre_bars"], top["post_bars"], 0)]
    controls: dict[str, dict[str, dict]] = {}
    for name, mode, flip in [
        ("direction_flip", top["mode"], True),
        ("symmetric_dot", "dot", False),
    ]:
        long_active, short_active, _ = signals(
            top_events,
            len(market),
            funding_q=top["funding_q"],
            score_q=top["score_q"],
            mode=mode,
            flip=flip,
        )
        controls[name] = simulate_all(market, dates, long_active, short_active, top["hold_bars"], extremes[top["hold_bars"]])
        print_stats(f"CONTROL {name}", controls[name])

    long_active, short_active = funding_only_signals(top_events, len(market), top["funding_q"])
    controls["funding_only"] = simulate_all(market, dates, long_active, short_active, top["hold_bars"], extremes[top["hold_bars"]])
    print_stats("CONTROL funding_only", controls["funding_only"])

    fake_events = build_event_features(
        market,
        dates,
        raw_events,
        pre_bars=top["pre_bars"],
        post_bars=top["post_bars"],
        event_offset_bars=48,
    )
    long_active, short_active, _ = signals(
        fake_events,
        len(market),
        funding_q=top["funding_q"],
        score_q=top["score_q"],
        mode=top["mode"],
    )
    controls["fake_settlement_plus_4h"] = simulate_all(market, dates, long_active, short_active, top["hold_bars"], extremes[top["hold_bars"]])
    print_stats("CONTROL fake_settlement_plus_4h", controls["fake_settlement_plus_4h"])

    serialisable_rows = []
    for row in rows:
        serialisable = dict(row)
        serialisable["score"] = list(row["score"])
        serialisable_rows.append(serialisable)
    Path("results/funding_settlement_curl_alpha_scan_2026-07-13.json").write_text(
        json.dumps(
            {
                "hypothesis": "antisymmetric pre/post premium-OI response around known funding settlements identifies trapped leveraged inventory",
                "causality": {
                    "cutoff": "2024-01-01 physical",
                    "funding_event": "exact funding_time, ceiled to next 5m bar",
                    "signal": "after post window; execution next bar open",
                    "oi": "one completed 5m source bar delayed by canonical loader",
                    "premium": "backward-asof completed 1h premium close",
                    "standardisation": "shift(1) prior-event rolling 180 events, min 60",
                },
                "rows": serialisable_rows,
                "controls": controls,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
