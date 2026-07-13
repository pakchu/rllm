from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from training.search_positioning_disagreement_alpha import _future_extreme, _simulate_no_stop
from training.search_positioning_hgb_path_alpha import _read_before

MARKET = "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz"
SPOT = "data/cache_spot_premium_5m_2020-01-01_2026-06-01.csv.gz"
WINDOWS = {
    "fit": ("2020-06-01", "2023-01-01"),
    "fit_2020_h2": ("2020-06-01", "2021-01-01"),
    "fit_2021_h1": ("2021-01-01", "2021-07-01"),
    "fit_2021_h2": ("2021-07-01", "2022-01-01"),
    "fit_2022_h1": ("2022-01-01", "2022-07-01"),
    "fit_2022_h2": ("2022-07-01", "2023-01-01"),
    "select_2023": ("2023-01-01", "2024-01-01"),
    "select_2023_h1": ("2023-01-01", "2023-07-01"),
    "select_2023_h2": ("2023-07-01", "2024-01-01"),
}
SEGMENTS = ["fit_2020_h2", "fit_2021_h1", "fit_2021_h2", "fit_2022_h1", "fit_2022_h2", "select_2023_h1", "select_2023_h2"]


def prior_z(values: pd.Series, window: int) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    prior = values.shift(1)
    mean = prior.rolling(window, min_periods=max(288, window // 2)).mean()
    std = prior.rolling(window, min_periods=max(288, window // 2)).std(ddof=0).replace(0, np.nan)
    return (values - mean) / std


def load_pre2024() -> tuple[pd.DataFrame, pd.Series]:
    market = _read_before(MARKET, "date", "2024-01-01")
    spot = _read_before(SPOT, "date", "2024-01-01")
    market["date"] = pd.to_datetime(market["date"], utc=True).dt.tz_convert(None)
    spot["date"] = pd.to_datetime(spot["date"], utc=True).dt.tz_convert(None)
    market = market.sort_values("date").drop_duplicates("date", keep="last")
    spot = spot.sort_values("date").drop_duplicates("date", keep="last")
    market = market.merge(spot[["date", "spot_close", "spot_rows"]], on="date", how="left", validate="one_to_one").reset_index(drop=True)
    dates = pd.to_datetime(market["date"])
    if dates.max() >= pd.Timestamp("2024-01-01"):
        raise RuntimeError("future rows opened")
    return market, dates


def discretize_returns(log_return: pd.Series, threshold: float) -> np.ndarray:
    prior_vol = log_return.shift(1).rolling(288, min_periods=144).std(ddof=0).replace(0, np.nan)
    z = (log_return / prior_vol).to_numpy(float)
    state = np.full(len(z), -1, dtype=np.int8)
    finite = np.isfinite(z)
    state[finite & (z < -threshold)] = 0
    state[finite & (np.abs(z) <= threshold)] = 1
    state[finite & (z > threshold)] = 2
    return state


def _entropy_from_counts(triple: np.ndarray, cond: np.ndarray, pair: np.ndarray, prev: np.ndarray) -> float:
    total = float(triple.sum())
    if total <= 0:
        return np.nan
    value = 0.0
    for ycur in range(3):
        for yprev in range(3):
            for xprev in range(3):
                c3 = triple[ycur, yprev, xprev]
                if c3 <= 0:
                    continue
                ccond = cond[yprev, xprev]
                cpair = pair[ycur, yprev]
                cprev = prev[yprev]
                if min(ccond, cpair, cprev) <= 0:
                    continue
                value += (c3 / total) * np.log((c3 * cprev) / (ccond * cpair))
    return float(value)


def rolling_transfer_entropy(
    x_state: np.ndarray,
    y_state: np.ndarray,
    *,
    window: int,
    decision: np.ndarray,
) -> np.ndarray:
    """TE X->Y from transitions ending strictly before each decision bar."""
    n = len(x_state)
    out = np.full(n, np.nan)
    triple = np.zeros((3, 3, 3), dtype=np.int64)
    cond = np.zeros((3, 3), dtype=np.int64)
    pair = np.zeros((3, 3), dtype=np.int64)
    prev = np.zeros(3, dtype=np.int64)

    def update(i: int, delta: int) -> None:
        if i <= 0:
            return
        xp, yp, yc = int(x_state[i - 1]), int(y_state[i - 1]), int(y_state[i])
        if min(xp, yp, yc) < 0:
            return
        triple[yc, yp, xp] += delta
        cond[yp, xp] += delta
        pair[yc, yp] += delta
        prev[yp] += delta

    for i in range(1, n):
        # At i, counts end at i-1 and therefore do not use the current state.
        if decision[i] and i > window // 2:
            out[i] = _entropy_from_counts(triple, cond, pair, prev)
        update(i, +1)
        expired = i - window
        if expired >= 1:
            update(expired, -1)
    return out


def build_features(market: pd.DataFrame, dates: pd.Series, *, state_threshold: float, te_window: int) -> pd.DataFrame:
    perp = np.log(pd.to_numeric(market["close"], errors="coerce").where(lambda x: x > 0))
    spot = np.log(pd.to_numeric(market["spot_close"], errors="coerce").where(lambda x: x > 0))
    complete = pd.to_numeric(market["spot_rows"], errors="coerce").eq(5)
    perp_ret = perp.diff().where(complete)
    spot_ret = spot.diff().where(complete)
    perp_state = discretize_returns(perp_ret, state_threshold)
    spot_state = discretize_returns(spot_ret, state_threshold)
    decision = (dates.dt.minute == 0).to_numpy(bool) & complete.to_numpy(bool)
    spot_to_perp = rolling_transfer_entropy(spot_state, perp_state, window=te_window, decision=decision)
    perp_to_spot = rolling_transfer_entropy(perp_state, spot_state, window=te_window, decision=decision)
    te_advantage = spot_to_perp - perp_to_spot
    spot_move = spot - spot.shift(12)
    perp_move = perp - perp.shift(12)
    lead_gap_z = prior_z(spot_move - perp_move, 2016)
    return pd.DataFrame(
        {
            "spot_to_perp_te": spot_to_perp,
            "perp_to_spot_te": perp_to_spot,
            "te_advantage": te_advantage,
            "lead_gap_z": lead_gap_z,
            "spot_move": spot_move,
            "decision": decision,
        }
    ).replace([np.inf, -np.inf], np.nan)


def fit_quantile(features: pd.DataFrame, dates: pd.Series, column: str, q: float, absolute: bool = False) -> float:
    a, b = WINDOWS["fit"]
    mask = (dates >= pd.Timestamp(a)) & (dates < pd.Timestamp(b)) & features["decision"]
    values = features.loc[mask, column].dropna()
    if absolute:
        values = values.abs()
    if len(values) < 5_000:
        raise ValueError(f"insufficient fit decisions {column}: {len(values)}")
    return float(values.quantile(q))


def signals(features: pd.DataFrame, te_threshold: float, gap_threshold: float, mode: str, *, flip: bool = False, no_te: bool = False, reverse_te: bool = False) -> tuple[np.ndarray, np.ndarray]:
    te = features["te_advantage"].to_numpy(float)
    gap = features["lead_gap_z"].to_numpy(float)
    spot_move = features["spot_move"].to_numpy(float)
    active = features["decision"].to_numpy(bool) & np.isfinite(gap) & (np.abs(gap) >= gap_threshold)
    if not no_te:
        active &= np.isfinite(te) & ((te <= -te_threshold) if reverse_te else (te >= te_threshold))
    if mode == "spot_aligned":
        active &= np.sign(gap) == np.sign(spot_move)
    elif mode != "gap_only":
        raise KeyError(mode)
    side = np.sign(gap)
    if flip:
        side = -side
    return active & (side > 0), active & (side < 0)


def simulate(market: pd.DataFrame, dates: pd.Series, la: np.ndarray, sa: np.ndarray, hold: int, extremes: tuple[np.ndarray, np.ndarray]) -> dict:
    return {w: _simulate_no_stop(market, dates, la, sa, window=w, hold_bars=hold, stride_bars=1, leverage=.5, fee_rate=.0005, slippage_rate=.0001, extremes=extremes, windows=WINDOWS) for w in WINDOWS}


def rank_key(stats: dict) -> tuple:
    enough=stats["fit"]["trades"]>=80 and stats["select_2023"]["trades"]>=24 and min(stats[w]["trades"] for w in ["select_2023_h1","select_2023_h2"])>=8
    core=[stats[w]["ratio"] for w in ["fit","select_2023","select_2023_h1","select_2023_h2"]]
    pos=sum(stats[w]["return_pct"]>0 for w in SEGMENTS)
    return enough,min(core)>0,pos,min(core),float(np.median(core)),stats["select_2023"]["trades"]


def print_stats(title: str, stats: dict) -> None:
    print("\n"+title)
    for w in ["fit","select_2023",*SEGMENTS]:
        s=stats[w];print(w,f"ret={s['return_pct']:.2f}",f"cagr={s['cagr_pct']:.2f}",f"mdd={s['strict_mdd_pct']:.2f}",f"ratio={s['ratio']:.2f}",f"n={s['trades']}",f"L/S={s['longs']}/{s['shorts']}")


def main() -> None:
    market,dates=load_pre2024();holds=[12,24,48];extremes={h:(_future_extreme(market.low.to_numpy(float),h,"min"),_future_extreme(market.high.to_numpy(float),h,"max")) for h in holds};rows=[];banks={}
    for stw,tew in itertools.product([.5,1.0],[2016,8640]):
        f=build_features(market,dates,state_threshold=stw,te_window=tew);banks[(stw,tew)]=f
        for teq,gq,hold,mode in itertools.product([.5,.7],[.8,.9],holds,["spot_aligned","gap_only"]):
            teth=fit_quantile(f,dates,"te_advantage",teq);gth=fit_quantile(f,dates,"lead_gap_z",gq,True);la,sa=signals(f,teth,gth,mode);st=simulate(market,dates,la,sa,hold,extremes[hold]);rows.append({"state_threshold":stw,"te_window":tew,"te_q":teq,"gap_q":gq,"hold":hold,"mode":mode,"te_threshold":teth,"gap_threshold":gth,"rank":rank_key(st),"stats":st})
    rows.sort(key=lambda r:r["rank"],reverse=True);print("candidates",len(rows))
    for i,r in enumerate(rows[:12],1):print_stats(f"RANK {i} st{r['state_threshold']} tw{r['te_window']} tq{r['te_q']} gq{r['gap_q']} h{r['hold']} {r['mode']} rank={r['rank']}",r['stats'])
    top=rows[0];f=banks[(top["state_threshold"],top["te_window"])];controls={}
    for name,kwargs in {"direction_flip":{"flip":True},"no_te":{"no_te":True},"reverse_te":{"reverse_te":True}}.items():
        la,sa=signals(f,top["te_threshold"],top["gap_threshold"],top["mode"],**kwargs);controls[name]=simulate(market,dates,la,sa,top["hold"],extremes[top["hold"]]);print_stats("CONTROL "+name,controls[name])
    Path("results/spot_perp_transfer_entropy_alpha_scan_2026-07-13.json").write_text(json.dumps({"rows":[{**r,"rank":list(r["rank"])} for r in rows],"controls":controls},indent=2))

if __name__=="__main__":main()
