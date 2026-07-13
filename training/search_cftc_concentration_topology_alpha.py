from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.search_cme_offshore_debt_handoff_alpha import prepare_cftc_reports, prior_z
from training.search_positioning_disagreement_alpha import _future_extreme, _simulate_no_stop
from training.search_positioning_hgb_path_alpha import _read_before

MARKET = "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz"
CFTC = "data/cftc_tff_cme_bitcoin_133741_2018_2026.csv.gz"
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


def load_pre2024() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    market = _read_before(MARKET, "date", "2024-01-01")
    market["date"] = pd.to_datetime(market["date"], utc=True).dt.tz_convert(None)
    market = market.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    dates = pd.to_datetime(market["date"])
    cftc = prepare_cftc_reports(pd.read_csv(CFTC, compression="infer"), cutoff="2024-01-01")
    if dates.max() >= pd.Timestamp("2024-01-01") or cftc.release_time.max() >= pd.Timestamp("2024-01-01"):
        raise RuntimeError("future source opened")
    return market, dates, cftc


def concentration_features(
    reports: pd.DataFrame,
    *,
    topology: str,
    breadth_horizon: int,
    swap_sides: bool = False,
    remove_breadth: bool = False,
) -> pd.DataFrame:
    out = reports.copy()
    l4 = pd.to_numeric(out["conc_net_le_4_tdr_long_all"], errors="coerce") / 100.0
    s4 = pd.to_numeric(out["conc_net_le_4_tdr_short_all"], errors="coerce") / 100.0
    l8 = pd.to_numeric(out["conc_net_le_8_tdr_long_all"], errors="coerce") / 100.0
    s8 = pd.to_numeric(out["conc_net_le_8_tdr_short_all"], errors="coerce") / 100.0
    if swap_sides:
        l4, s4, l8, s8 = s4, l4, s8, l8
    if topology == "rank_odds":
        eps = 1e-4
        raw = np.log((l4 + eps) / (l8 - l4 + eps)) - np.log((s4 + eps) / (s8 - s4 + eps))
    elif topology == "rank_curvature":
        raw = (2.0 * l4 - l8) - (2.0 * s4 - s8)
    else:
        raise KeyError(topology)
    topology_z = prior_z(raw, 104, 52)
    breadth = np.log(pd.to_numeric(out["traders_tot_all"], errors="coerce").where(lambda x: x > 0.0))
    breadth_change = breadth - breadth.shift(breadth_horizon)
    contraction_z = -prior_z(breadth_change, 104, 52)
    breadth_multiplier = 1.0 if remove_breadth else 1.0 + np.clip(contraction_z, 0.0, 3.0)
    out["topology_raw"] = raw
    out["topology_z"] = topology_z
    out["breadth_contraction_z"] = contraction_z
    out["fragility"] = np.abs(topology_z) * breadth_multiplier
    out["concentrated_side"] = np.sign(topology_z)
    return out


def fit_threshold(features: pd.DataFrame, quantile: float) -> float:
    start, end = WINDOWS["fit"]
    values = features.loc[(features.release_time >= start) & (features.release_time < end), "fragility"].dropna()
    if len(values) < 100:
        raise ValueError(f"insufficient fit reports: {len(values)}")
    return float(values.quantile(quantile))


def release_signals(
    features: pd.DataFrame,
    dates: pd.Series,
    threshold: float,
    mapping: str,
    *,
    release_extra_weeks: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    date_values = dates.to_numpy(dtype="datetime64[ns]")
    long_active = np.zeros(len(dates), dtype=bool)
    short_active = np.zeros(len(dates), dtype=bool)
    selected = features.loc[(features["fragility"] >= threshold) & (features["concentrated_side"] != 0.0)]
    for row in selected.itertuples(index=False):
        release = pd.Timestamp(row.release_time) + pd.Timedelta(weeks=release_extra_weeks)
        position = int(np.searchsorted(date_values, np.datetime64(release), side="left"))
        if position >= len(dates):
            continue
        concentrated_side = int(np.sign(row.concentrated_side))
        side = -concentrated_side if mapping == "fade" else concentrated_side
        long_active[position] = side > 0
        short_active[position] = side < 0
    return long_active, short_active


def simulate(
    market: pd.DataFrame,
    dates: pd.Series,
    long_active: np.ndarray,
    short_active: np.ndarray,
    hold: int,
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
            hold_bars=hold,
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
        stats["fit"]["trades"] >= 25
        and stats["select_2023"]["trades"] >= 8
        and min(stats["select_2023_h1"]["trades"], stats["select_2023_h2"]["trades"]) >= 3
    )
    return bool(
        enough
        and stats["fit"]["return_pct"] > 0
        and stats["fit"]["ratio"] >= 3
        and stats["select_2023"]["return_pct"] > 0
        and stats["select_2023"]["ratio"] >= 3
        and stats["select_2023_h1"]["return_pct"] >= 0
        and stats["select_2023_h2"]["return_pct"] >= 0
    )


def rank_key(stats: dict[str, dict[str, Any]]) -> tuple[Any, ...]:
    enough = (
        stats["fit"]["trades"] >= 25
        and stats["select_2023"]["trades"] >= 8
        and min(stats["select_2023_h1"]["trades"], stats["select_2023_h2"]["trades"]) >= 3
    )
    core = [stats[name]["ratio"] for name in ("fit", "select_2023", "select_2023_h1", "select_2023_h2")]
    return (
        admission(stats),
        enough,
        min(core) > 0,
        sum(stats[name]["return_pct"] > 0 for name in SEGMENTS),
        min(core),
        float(np.median(core)),
        stats["select_2023"]["trades"],
    )


def print_stats(title: str, stats: dict[str, dict[str, Any]]) -> None:
    print("\n" + title)
    for window in ("fit", "select_2023", *SEGMENTS):
        stats_window = stats[window]
        print(
            window,
            f"ret={stats_window['return_pct']:.2f}",
            f"cagr={stats_window['cagr_pct']:.2f}",
            f"mdd={stats_window['strict_mdd_pct']:.2f}",
            f"ratio={stats_window['ratio']:.2f}",
            f"n={stats_window['trades']}",
            f"L/S={stats_window['longs']}/{stats_window['shorts']}",
        )


def main() -> None:
    market, dates, reports = load_pre2024()
    holds = (576, 1152)
    extremes = {
        hold: (
            _future_extreme(market.low.to_numpy(float), hold, "min"),
            _future_extreme(market.high.to_numpy(float), hold, "max"),
        )
        for hold in holds
    }
    banks: dict[tuple[str, int], pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    grid = itertools.product(
        ("rank_odds", "rank_curvature"),
        (4, 13),
        (0.50, 0.70),
        ("fade", "follow"),
        holds,
    )
    for topology, breadth_horizon, tail, mapping, hold in grid:
        features = banks.setdefault(
            (topology, breadth_horizon),
            concentration_features(reports, topology=topology, breadth_horizon=breadth_horizon),
        )
        threshold = fit_threshold(features, tail)
        long_active, short_active = release_signals(features, dates, threshold, mapping)
        stats = simulate(market, dates, long_active, short_active, hold, extremes[hold])
        rows.append(
            {
                "topology": topology,
                "breadth_horizon": breadth_horizon,
                "tail_quantile": tail,
                "mapping": mapping,
                "hold": hold,
                "threshold": threshold,
                "rank": rank_key(stats),
                "stats": stats,
            }
        )
    rows.sort(key=lambda row: row["rank"], reverse=True)
    print("reports", len(reports), "candidates", len(rows), "admitted", sum(admission(row["stats"]) for row in rows))
    for index, row in enumerate(rows[:12], 1):
        print_stats(
            f"RANK {index} {row['topology']} b{row['breadth_horizon']} q{row['tail_quantile']} {row['mapping']} h{row['hold']} rank={row['rank']}",
            row["stats"],
        )

    top = rows[0]
    features = banks[(top["topology"], top["breadth_horizon"])]
    threshold = top["threshold"]
    hold = top["hold"]
    controls: dict[str, dict[str, dict[str, Any]]] = {}
    opposite_mapping = "follow" if top["mapping"] == "fade" else "fade"
    long_active, short_active = release_signals(features, dates, threshold, opposite_mapping)
    controls["direction_flip"] = simulate(market, dates, long_active, short_active, hold, extremes[hold])

    no_breadth = concentration_features(
        reports,
        topology=top["topology"],
        breadth_horizon=top["breadth_horizon"],
        remove_breadth=True,
    )
    no_breadth_threshold = fit_threshold(no_breadth, top["tail_quantile"])
    long_active, short_active = release_signals(no_breadth, dates, no_breadth_threshold, top["mapping"])
    controls["remove_breadth"] = simulate(market, dates, long_active, short_active, hold, extremes[hold])

    swapped = concentration_features(
        reports,
        topology=top["topology"],
        breadth_horizon=top["breadth_horizon"],
        swap_sides=True,
    )
    swapped_threshold = fit_threshold(swapped, top["tail_quantile"])
    long_active, short_active = release_signals(swapped, dates, swapped_threshold, top["mapping"])
    controls["swap_long_short_fields"] = simulate(market, dates, long_active, short_active, hold, extremes[hold])

    long_active, short_active = release_signals(
        features,
        dates,
        threshold,
        top["mapping"],
        release_extra_weeks=4,
    )
    controls["release_extra_4w"] = simulate(market, dates, long_active, short_active, hold, extremes[hold])
    for name, stats in controls.items():
        print_stats("CONTROL " + name, stats)

    base_long, base_short = release_signals(features, dates, threshold, top["mapping"])
    cost_stress = {
        str(side_bp): simulate(
            market,
            dates,
            base_long,
            base_short,
            hold,
            extremes[hold],
            side_cost=side_bp / 10_000,
        )
        for side_bp in (0, 1, 3, 6)
    }
    for side_bp, stats in cost_stress.items():
        print_stats(f"COST {side_bp}BP_SIDE", stats)

    result = {
        "protocol": {
            "source_cutoff": "strictly before 2024-01-01",
            "release": "report date +8d",
            "grid_size": 32,
            "entry": "first 5m decision bar at/after release; execute next 5m open",
            "leverage": 0.5,
            "cost": "6bp/side implementation cost",
            "strict_mdd": "favorable-first adverse-second OHLC high-water",
            "admission": "fit and 2023 CAGR/MDD >=3, positive 2023 halves, minimum 25/8/3 trades",
            "oos_opened": False,
        },
        "rows": [{**row, "rank": list(row["rank"])} for row in rows],
        "controls": controls,
        "cost_stress": cost_stress,
        "final_admitted": bool(
            admission(top["stats"]) and not any(admission(stats) for stats in controls.values())
        ),
    }
    Path("results/cftc_concentration_topology_alpha_scan_2026-07-13.json").write_text(
        json.dumps(result, indent=2)
    )


if __name__ == "__main__":
    main()
