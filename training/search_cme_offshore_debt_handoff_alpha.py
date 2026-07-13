"""Pre-2024 CME-to-offshore leveraged-debt handoff alpha preflight."""
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
CFTC = "data/cftc_tff_cme_bitcoin_133741_2018_2026.csv.gz"
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


def prior_z(values: pd.Series, window: int, minimum: int) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    prior = values.shift(1)
    return (values - prior.rolling(window, min_periods=minimum).mean()) / prior.rolling(window, min_periods=minimum).std(ddof=0).replace(0.0, np.nan)


def prepare_cftc_reports(cftc: pd.DataFrame, cutoff: str = "2024-01-01") -> pd.DataFrame:
    out = cftc.copy()
    out["report_date"] = pd.to_datetime(
        out["report_date_as_yyyy_mm_dd"], utc=True
    ).dt.tz_convert(None)
    # Deliberately conservative: no report is usable until report date +8d.
    out["release_time"] = out["report_date"] + pd.Timedelta(days=8)
    return (
        out.loc[out["release_time"] < pd.Timestamp(cutoff)]
        .sort_values("release_time")
        .reset_index(drop=True)
    )


def load_pre2024() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    market = _read_before(MARKET, "date", "2024-01-01")
    metrics = _read_before(METRICS, "create_time", "2024-01-01")
    market = _attach_delayed_metrics(market, metrics, tolerance="5min", delay_bars=1)
    market["date"] = pd.to_datetime(market["date"], utc=True).dt.tz_convert(None)
    dates = pd.to_datetime(market["date"])
    cftc = prepare_cftc_reports(pd.read_csv(CFTC, compression="infer"))
    if dates.max() >= pd.Timestamp("2024-01-01") or cftc.release_time.max() >= pd.Timestamp("2024-01-01"):
        raise RuntimeError("future source opened")
    return market, dates, cftc


def owner_state(market: pd.DataFrame) -> pd.Series:
    global_ratio = np.log(pd.to_numeric(market["count_long_short_ratio"], errors="coerce").where(lambda x: x > 0))
    quote = pd.to_numeric(market["quote_asset_volume"], errors="coerce")
    taker_buy = pd.to_numeric(market["taker_buy_quote"], errors="coerce")
    flow = ((2 * taker_buy / quote.replace(0, np.nan)) - 1).rolling(12, min_periods=12).mean()
    return np.tanh(0.5 * prior_z(global_ratio, 288, 144) + 0.5 * prior_z(flow, 288, 144))


def build_events(market: pd.DataFrame, dates: pd.Series, cftc: pd.DataFrame, *, offshore_window: int, cftc_window: int, cftc_source: str = "leveraged", release_extra_weeks: int = 0) -> pd.DataFrame:
    oi = np.log(pd.to_numeric(market["sum_open_interest"], errors="coerce").where(lambda x: x > 0))
    owner = owner_state(market)
    offshore = (oi - oi.shift(offshore_window)) * (owner - owner.shift(offshore_window))
    offshore_z = prior_z(offshore, 8640, 2016)

    if cftc_source == "asset_mgr":
        net = (pd.to_numeric(cftc["asset_mgr_positions_long"], errors="coerce") - pd.to_numeric(cftc["asset_mgr_positions_short"], errors="coerce")) / pd.to_numeric(cftc["open_interest_all"], errors="coerce")
    else:
        net = (pd.to_numeric(cftc["lev_money_positions_long"], errors="coerce") - pd.to_numeric(cftc["lev_money_positions_short"], errors="coerce")) / pd.to_numeric(cftc["open_interest_all"], errors="coerce")
    dc = net.diff()
    dc_z = prior_z(dc, cftc_window, max(26, cftc_window // 2))
    date_values = dates.to_numpy(dtype="datetime64[ns]")
    rows = []
    for i, row in cftc.iterrows():
        release = pd.Timestamp(row.release_time) + pd.Timedelta(weeks=release_extra_weeks)
        pos = int(np.searchsorted(date_values, np.datetime64(release), side="left"))
        if pos >= len(market) or not (np.isfinite(dc_z.iloc[i]) and np.isfinite(offshore_z.iloc[pos]) and np.isfinite(offshore.iloc[pos])):
            continue
        handoff = -float(dc_z.iloc[i]) * float(offshore_z.iloc[pos])
        rows.append({"release_time": release, "signal_pos": pos, "handoff": handoff, "offshore": float(offshore.iloc[pos]), "dc_z": float(dc_z.iloc[i]), "offshore_z": float(offshore_z.iloc[pos])})
    return pd.DataFrame(rows)


def fit_threshold(events: pd.DataFrame, q: float, column: str = "handoff") -> float:
    a, b = WINDOWS["fit"]
    values = events.loc[(events.release_time >= a) & (events.release_time < b), column].dropna()
    if len(values) < 60:
        raise ValueError(len(values))
    return float(values.quantile(q))


def signals(events: pd.DataFrame, n: int, threshold: float, *, flip: bool = False, offshore_only: bool = False) -> tuple[np.ndarray, np.ndarray]:
    score = events["handoff"] if not offshore_only else events["offshore_z"].abs()
    selected = events.loc[score >= threshold]
    side = -np.sign(selected["offshore"].to_numpy(float))
    if flip:
        side = -side
    pos = selected.signal_pos.to_numpy(int)
    la = np.zeros(n, bool); sa = np.zeros(n, bool)
    la[pos[side > 0]] = True; sa[pos[side < 0]] = True
    return la, sa


def simulate(market: pd.DataFrame, dates: pd.Series, la: np.ndarray, sa: np.ndarray, hold: int, extremes: tuple[np.ndarray, np.ndarray]) -> dict:
    return {w: _simulate_no_stop(market, dates, la, sa, window=w, hold_bars=hold, stride_bars=1, leverage=.5, fee_rate=.0005, slippage_rate=.0001, extremes=extremes, windows=WINDOWS) for w in WINDOWS}


def rank_key(stats: dict) -> tuple:
    enough = stats["fit"]["trades"] >= 20 and stats["select_2023"]["trades"] >= 8 and min(stats[w]["trades"] for w in ["select_2023_h1", "select_2023_h2"]) >= 3
    pos = sum(stats[w]["return_pct"] > 0 for w in SEGMENTS)
    core = [stats[w]["ratio"] for w in ["fit", "select_2023", "select_2023_h1", "select_2023_h2"]]
    return enough, min(core) > 0.0, pos, min(core), float(np.median(core)), stats["select_2023"]["trades"]


def print_stats(title: str, stats: dict) -> None:
    print("\n" + title)
    for w in ["fit", "select_2023", *SEGMENTS]:
        s=stats[w]; print(w,f"ret={s['return_pct']:.2f}",f"cagr={s['cagr_pct']:.2f}",f"mdd={s['strict_mdd_pct']:.2f}",f"ratio={s['ratio']:.2f}",f"n={s['trades']}",f"L/S={s['longs']}/{s['shorts']}")


def main() -> None:
    market,dates,cftc=load_pre2024(); holds=[288,576]; extremes={h:(_future_extreme(market.low.to_numpy(float),h,"min"),_future_extreme(market.high.to_numpy(float),h,"max")) for h in holds}; rows=[]; banks={}
    for ow,cw,q,h in itertools.product([2016,4032],[52,104],[.8,.9],holds):
        e=banks.setdefault((ow,cw,"leveraged",0),build_events(market,dates,cftc,offshore_window=ow,cftc_window=cw)); th=fit_threshold(e,q); la,sa=signals(e,len(market),th); st=simulate(market,dates,la,sa,h,extremes[h]); rows.append({"offshore_window":ow,"cftc_window":cw,"q":q,"hold":h,"threshold":th,"rank":rank_key(st),"stats":st})
    rows.sort(key=lambda r:r["rank"],reverse=True); print("reports",len(cftc),"candidates",len(rows))
    for i,r in enumerate(rows,1): print_stats(f"RANK {i} ow{r['offshore_window']} cw{r['cftc_window']} q{r['q']} h{r['hold']} rank={r['rank']}",r['stats'])
    top=rows[0]; controls={}
    specs={"direction_flip":("leveraged",0,False,True),"asset_mgr":("asset_mgr",0,False,False),"cftc_extra_4w":("leveraged",4,False,False),"offshore_only":("leveraged",0,True,False)}
    for name,(src,lag,only,flip) in specs.items():
        e=build_events(market,dates,cftc,offshore_window=top["offshore_window"],cftc_window=top["cftc_window"],cftc_source=src,release_extra_weeks=lag)
        th=fit_threshold(e,top["q"]) if not only else float(e.loc[(e.release_time>=WINDOWS['fit'][0])&(e.release_time<WINDOWS['fit'][1]),"offshore_z"].abs().quantile(top["q"]))
        la,sa=signals(e,len(market),th,flip=flip,offshore_only=only); controls[name]=simulate(market,dates,la,sa,top["hold"],extremes[top["hold"]]); print_stats("CONTROL "+name,controls[name])
    Path("results/cme_offshore_debt_handoff_alpha_scan_2026-07-13.json").write_text(json.dumps({"rows":[{**r,"rank":list(r["rank"])} for r in rows],"controls":controls},indent=2))

if __name__=="__main__": main()
