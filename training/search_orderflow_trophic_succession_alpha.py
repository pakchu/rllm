from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.search_positioning_disagreement_alpha import _future_extreme, _simulate_no_stop
from training.search_positioning_hgb_path_alpha import _read_before
from training.search_spot_perp_transfer_entropy_alpha import prior_z

MARKET = "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz"
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
SEGMENTS = (
    "fit_2020_h2",
    "fit_2021_h1",
    "fit_2021_h2",
    "fit_2022_h1",
    "fit_2022_h2",
    "select_2023_h1",
    "select_2023_h2",
)
PROFILES = (
    (3, 3, 2),
    (3, 6, 3),
    (6, 6, 3),
    (6, 12, 6),
    (12, 12, 6),
    (12, 24, 6),
)


def load_pre2024() -> tuple[pd.DataFrame, pd.Series]:
    market = _read_before(MARKET, "date", "2024-01-01")
    market["date"] = pd.to_datetime(market["date"], utc=True).dt.tz_convert(None)
    market = market.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    dates = pd.to_datetime(market["date"])
    if dates.max() >= pd.Timestamp("2024-01-01"):
        raise RuntimeError("future rows opened")
    return market, dates


def _phase_sum(values: pd.Series, length: int, end_shift: int) -> pd.Series:
    return values.shift(end_shift).rolling(length, min_periods=length).sum()


def _phase_extreme(values: pd.Series, length: int, end_shift: int, kind: str) -> pd.Series:
    rolling = values.shift(end_shift).rolling(length, min_periods=length)
    return rolling.max() if kind == "max" else rolling.min()


def build_phase(market: pd.DataFrame, *, length: int, end_shift: int, prefix: str) -> pd.DataFrame:
    quote = pd.to_numeric(market["quote_asset_volume"], errors="coerce")
    trades = pd.to_numeric(market["number_of_trades"], errors="coerce")
    taker_buy = pd.to_numeric(market["taker_buy_quote"], errors="coerce")
    signed = 2.0 * taker_buy - quote
    close = np.log(pd.to_numeric(market["close"], errors="coerce").where(lambda x: x > 0.0))
    high = np.log(pd.to_numeric(market["high"], errors="coerce").where(lambda x: x > 0.0))
    low = np.log(pd.to_numeric(market["low"], errors="coerce").where(lambda x: x > 0.0))

    quote_sum = _phase_sum(quote, length, end_shift)
    trade_sum = _phase_sum(trades, length, end_shift)
    signed_sum = _phase_sum(signed, length, end_shift)
    imbalance = signed_sum / quote_sum.replace(0.0, np.nan)
    ticket = quote_sum / trade_sum.replace(0.0, np.nan)
    intensity = trade_sum / float(length)
    phase_return = close.shift(end_shift) - close.shift(end_shift + length)
    phase_high = _phase_extreme(high, length, end_shift, "max")
    phase_low = _phase_extreme(low, length, end_shift, "min")
    close_location = 2.0 * (close.shift(end_shift) - phase_low) / (phase_high - phase_low).replace(0.0, np.nan) - 1.0
    ticket_z = prior_z(np.log(ticket.where(ticket > 0.0)), 2016)
    intensity_z = prior_z(np.log(intensity.where(intensity > 0.0)), 2016)
    return_z = prior_z(phase_return, 2016)
    impact = return_z.abs() / (imbalance.abs() + 0.02)
    return pd.DataFrame(
        {
            f"{prefix}_imbalance": imbalance,
            f"{prefix}_imbalance_z": prior_z(imbalance, 2016),
            f"{prefix}_ticket_z": ticket_z,
            f"{prefix}_intensity_z": intensity_z,
            f"{prefix}_return_z": return_z,
            f"{prefix}_impact": impact,
            f"{prefix}_impact_z": prior_z(impact, 2016),
            f"{prefix}_clv": close_location,
        }
    ).replace([np.inf, -np.inf], np.nan)


def build_profile_features(market: pd.DataFrame, profile: tuple[int, int, int]) -> pd.DataFrame:
    sponsor_bars, crowd_bars, absorb_bars = profile
    absorb = build_phase(market, length=absorb_bars, end_shift=0, prefix="a")
    crowd = build_phase(market, length=crowd_bars, end_shift=absorb_bars, prefix="c")
    sponsor = build_phase(market, length=sponsor_bars, end_shift=absorb_bars + crowd_bars, prefix="s")
    return pd.concat([sponsor, crowd, absorb], axis=1)


def fit_quantile(values: pd.Series | np.ndarray, dates: pd.Series, quantile: float) -> float:
    start, end = WINDOWS["fit"]
    mask = (dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))
    series = pd.Series(values, index=dates.index)
    sample = pd.to_numeric(series[mask], errors="coerce").dropna()
    if len(sample) < 5_000:
        raise ValueError(f"insufficient fit sample: {len(sample)}")
    return float(sample.quantile(quantile))


def role_scores(
    features: pd.DataFrame,
    *,
    order_swap: bool = False,
    ticket_role_swap: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sponsor_prefix, crowd_prefix = ("c", "s") if order_swap else ("s", "c")
    sponsor_imbalance = features[f"{sponsor_prefix}_imbalance"].to_numpy(float)
    direction = np.sign(sponsor_imbalance)
    sponsor_flow = direction * features[f"{sponsor_prefix}_imbalance_z"].to_numpy(float)
    sponsor_ticket = features[f"{sponsor_prefix}_ticket_z"].to_numpy(float)
    sponsor_intensity = features[f"{sponsor_prefix}_intensity_z"].to_numpy(float)
    sponsor_progress = direction * features[f"{sponsor_prefix}_return_z"].to_numpy(float)
    sponsor_impact = features[f"{sponsor_prefix}_impact"].to_numpy(float)

    crowd_flow = direction * features[f"{crowd_prefix}_imbalance_z"].to_numpy(float)
    crowd_ticket = features[f"{crowd_prefix}_ticket_z"].to_numpy(float)
    crowd_intensity = features[f"{crowd_prefix}_intensity_z"].to_numpy(float)
    crowd_progress = direction * features[f"{crowd_prefix}_return_z"].to_numpy(float)
    crowd_impact = features[f"{crowd_prefix}_impact"].to_numpy(float)
    impact_decay = np.log((crowd_impact + 0.10) / (sponsor_impact + 0.10))

    if ticket_role_swap:
        sponsor_role = sponsor_flow - sponsor_ticket + sponsor_intensity + sponsor_progress
        crowd_role = crowd_flow - crowd_intensity + crowd_ticket + crowd_progress - impact_decay
    else:
        sponsor_role = sponsor_flow + sponsor_ticket - sponsor_intensity + sponsor_progress
        crowd_role = crowd_flow + crowd_intensity - crowd_ticket + crowd_progress - impact_decay
    absorption_role = (
        direction * features["a_imbalance_z"].to_numpy(float)
        - direction * features["a_return_z"].to_numpy(float)
        - features["a_impact_z"].to_numpy(float)
        - direction * features["a_clv"].to_numpy(float)
    )
    return direction, sponsor_role, crowd_role, absorption_role


def fit_policy_thresholds(features: pd.DataFrame, dates: pd.Series, tail_quantile: float) -> dict[str, float]:
    _, sponsor_role, crowd_role, absorption_role = role_scores(features)
    return {
        "tail_quantile": tail_quantile,
        "sponsor_role": fit_quantile(sponsor_role, dates, tail_quantile),
        "crowd_role": fit_quantile(crowd_role, dates, tail_quantile),
        "absorption_role": fit_quantile(absorption_role, dates, tail_quantile),
    }


def sequence_signals(
    features: pd.DataFrame,
    thresholds: dict[str, float],
    branch: str,
    *,
    flip: bool = False,
    order_swap: bool = False,
    ticket_role_swap: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    direction, sponsor_role, crowd_role, absorption_role = role_scores(
        features,
        order_swap=order_swap,
        ticket_role_swap=ticket_role_swap,
    )
    sponsor = np.isfinite(sponsor_role) & (direction != 0.0) & (sponsor_role >= thresholds["sponsor_role"])
    sponsor_crowd = sponsor & np.isfinite(crowd_role) & (crowd_role >= thresholds["crowd_role"])
    absorbed = sponsor_crowd & np.isfinite(absorption_role) & (absorption_role >= thresholds["absorption_role"])
    if branch == "continuation":
        active = sponsor_crowd & ~absorbed
        side = direction
    elif branch == "absorption_reversal":
        active = absorbed
        side = -direction
    elif branch == "sponsor_only":
        active = sponsor
        side = direction
    else:
        raise KeyError(branch)
    onset = active & ~np.roll(active, 1)
    onset[0] = False
    if flip:
        side = -side
    long_active = onset & (side > 0.0)
    short_active = onset & (side < 0.0)
    return long_active, short_active, {"sponsor": sponsor, "crowd": sponsor_crowd, "absorbed": absorbed, "active": onset}


def simulate(
    market: pd.DataFrame,
    dates: pd.Series,
    long_active: np.ndarray,
    short_active: np.ndarray,
    hold_bars: int,
    extremes: tuple[np.ndarray, np.ndarray],
    side_cost: float = 0.0006,
) -> dict[str, dict[str, Any]]:
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
            fee_rate=side_cost,
            slippage_rate=0.0,
            extremes=extremes,
            windows=WINDOWS,
        )
        for window in WINDOWS
    }


def admission(stats: dict[str, dict[str, Any]]) -> bool:
    enough = (
        stats["fit"]["trades"] >= 80
        and stats["select_2023"]["trades"] >= 20
        and min(stats["select_2023_h1"]["trades"], stats["select_2023_h2"]["trades"]) >= 6
        and stats["fit"]["longs"] > 0
        and stats["fit"]["shorts"] > 0
        and stats["select_2023"]["longs"] > 0
        and stats["select_2023"]["shorts"] > 0
    )
    return bool(
        enough
        and stats["fit"]["return_pct"] > 0.0
        and stats["fit"]["ratio"] >= 2.0
        and stats["select_2023"]["return_pct"] > 0.0
        and stats["select_2023"]["ratio"] >= 1.25
        and stats["select_2023_h1"]["return_pct"] >= 0.0
        and stats["select_2023_h2"]["return_pct"] >= 0.0
    )


def rank_key(stats: dict[str, dict[str, Any]]) -> tuple[Any, ...]:
    enough = stats["fit"]["trades"] >= 80 and stats["select_2023"]["trades"] >= 20
    core = [stats[name]["ratio"] for name in ("fit", "select_2023", "select_2023_h1", "select_2023_h2")]
    positive_segments = sum(stats[name]["return_pct"] > 0.0 for name in SEGMENTS)
    return (
        admission(stats),
        enough,
        min(core) > 0.0,
        positive_segments,
        min(core),
        float(np.median(core)),
        stats["select_2023"]["trades"],
    )


def print_stats(title: str, stats: dict[str, dict[str, Any]]) -> None:
    print("\n" + title)
    for window in ("fit", "select_2023", *SEGMENTS):
        value = stats[window]
        print(window, f"ret={value['return_pct']:.2f}", f"cagr={value['cagr_pct']:.2f}", f"mdd={value['strict_mdd_pct']:.2f}", f"ratio={value['ratio']:.2f}", f"n={value['trades']}", f"L/S={value['longs']}/{value['shorts']}")


def main() -> None:
    market, dates = load_pre2024()
    holds = (24, 72)
    extremes = {
        hold: (
            _future_extreme(market["low"].to_numpy(float), hold, "min"),
            _future_extreme(market["high"].to_numpy(float), hold, "max"),
        )
        for hold in holds
    }
    rows: list[dict[str, Any]] = []
    banks: dict[tuple[tuple[int, int, int], float], tuple[pd.DataFrame, dict[str, float]]] = {}
    for profile, tail_quantile in itertools.product(PROFILES, (0.70, 0.80, 0.90, 0.95)):
        features = build_profile_features(market, profile)
        thresholds = fit_policy_thresholds(features, dates, tail_quantile)
        banks[(profile, tail_quantile)] = (features, thresholds)
        for branch, hold in itertools.product(("continuation", "absorption_reversal"), holds):
            long_active, short_active, diagnostics = sequence_signals(features, thresholds, branch)
            stats = simulate(market, dates, long_active, short_active, hold, extremes[hold])
            rows.append(
                {
                    "profile": list(profile),
                    "tail_quantile": tail_quantile,
                    "branch": branch,
                    "hold": hold,
                    "thresholds": thresholds,
                    "sponsor_events": int(diagnostics["sponsor"].sum()),
                    "crowd_events": int(diagnostics["crowd"].sum()),
                    "absorption_events": int(diagnostics["absorbed"].sum()),
                    "raw_events": int(diagnostics["active"].sum()),
                    "prelim_admitted": admission(stats),
                    "rank": rank_key(stats),
                    "stats": stats,
                }
            )
    rows.sort(key=lambda row: row["rank"], reverse=True)
    print("candidates", len(rows), "prelim_admitted", sum(row["prelim_admitted"] for row in rows))
    for index, row in enumerate(rows[:12], 1):
        print_stats(f"RANK {index} profile={row['profile']} {row['branch']} h{row['hold']} events={row['raw_events']} S/C/A={row['sponsor_events']}/{row['crowd_events']}/{row['absorption_events']} rank={row['rank']}", row["stats"])

    top = rows[0]
    profile = tuple(top["profile"])
    features, thresholds = banks[(profile, top["tail_quantile"])]
    base_long, base_short, _ = sequence_signals(features, thresholds, top["branch"])
    controls: dict[str, dict[str, dict[str, Any]]] = {}
    for name, kwargs in (
        ("direction_flip", {"flip": True}),
        ("phase_order_swap", {"order_swap": True}),
        ("ticket_role_swap", {"ticket_role_swap": True}),
    ):
        long_active, short_active, _ = sequence_signals(features, thresholds, top["branch"], **kwargs)
        controls[name] = simulate(market, dates, long_active, short_active, top["hold"], extremes[top["hold"]])
    sponsor_long, sponsor_short, _ = sequence_signals(features, thresholds, "sponsor_only")
    controls["sponsor_only"] = simulate(market, dates, sponsor_long, sponsor_short, top["hold"], extremes[top["hold"]])
    lag = sum(profile)
    lag_long = np.r_[np.zeros(lag, dtype=bool), base_long[:-lag]]
    lag_short = np.r_[np.zeros(lag, dtype=bool), base_short[:-lag]]
    controls["full_sequence_lag"] = simulate(market, dates, lag_long, lag_short, top["hold"], extremes[top["hold"]])
    for name, stats in controls.items():
        print_stats("CONTROL " + name, stats)

    cost_stress = {
        str(side_bp): simulate(
            market,
            dates,
            base_long,
            base_short,
            top["hold"],
            extremes[top["hold"]],
            side_cost=side_bp / 10_000.0,
        )
        for side_bp in (0, 1, 3, 6)
    }
    for side_bp, stats in cost_stress.items():
        print_stats(f"COST {side_bp}BP_SIDE", stats)

    control_pass = not any(admission(stats) for stats in controls.values())
    final_admitted = bool(top["prelim_admitted"] and control_pass)
    result = {
        "protocol": {
            "source_cutoff": "strictly before 2024-01-01",
            "fit_threshold_window": WINDOWS["fit"],
            "grid_size": 96,
            "phase_profiles": [list(profile) for profile in PROFILES],
            "entry": "next 5m open",
            "leverage": 0.5,
            "cost": "6bp/side implementation cost",
            "strict_mdd": "favorable-first adverse-second OHLC high-water",
            "oos_opened": False,
        },
        "rows": [{**row, "rank": list(row["rank"])} for row in rows],
        "controls": controls,
        "cost_stress": cost_stress,
        "top_control_pass": control_pass,
        "final_admitted": final_admitted,
    }
    Path("results/orderflow_trophic_succession_alpha_scan_2026-07-13.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
