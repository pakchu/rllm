"""Search BTC alpha in CFTC position surprises not yet assimilated by price."""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.search_cftc_concentration_topology_alpha import (
    WINDOWS,
    admission,
    load_pre2024,
    print_stats,
    rank_key,
    simulate,
)
from training.search_cme_offshore_debt_handoff_alpha import prior_z
from training.search_positioning_disagreement_alpha import _future_extreme

PARTICIPANT_COLUMNS = {
    "leveraged_money": ("lev_money_positions_long", "lev_money_positions_short"),
    "asset_manager": ("asset_mgr_positions_long", "asset_mgr_positions_short"),
}


def participant_surprise(reports: pd.DataFrame, participant: str) -> pd.Series:
    long_column, short_column = PARTICIPANT_COLUMNS[participant]
    open_interest = pd.to_numeric(reports["open_interest_all"], errors="coerce")
    net_share = (
        pd.to_numeric(reports[long_column], errors="coerce")
        - pd.to_numeric(reports[short_column], errors="coerce")
    ) / open_interest.replace(0.0, np.nan)
    return prior_z(net_share.diff(), 104, 52)


def build_assimilation_events(
    market: pd.DataFrame,
    dates: pd.Series,
    reports: pd.DataFrame,
    *,
    participant: str,
) -> pd.DataFrame:
    surprise = participant_surprise(reports, participant)
    date_values = dates.to_numpy(dtype="datetime64[ns]")
    close = pd.to_numeric(market["close"], errors="coerce").to_numpy(float)
    rows: list[dict[str, Any]] = []
    for index, report in reports.iterrows():
        report_position = int(
            np.searchsorted(date_values, np.datetime64(report.report_date), side="left")
        )
        release_position = int(
            np.searchsorted(date_values, np.datetime64(report.release_time), side="left")
        )
        if (
            report_position >= len(market)
            or release_position >= len(market)
            or release_position <= report_position
            or not np.isfinite(surprise.iloc[index])
            or close[report_position] <= 0.0
            or close[release_position] <= 0.0
        ):
            continue
        rows.append(
            {
                "report_date": report.report_date,
                "release_time": report.release_time,
                "signal_position": release_position,
                "position_surprise_z": float(surprise.iloc[index]),
                "report_to_release_return": float(
                    np.log(close[release_position] / close[report_position])
                ),
            }
        )
    events = pd.DataFrame(rows)
    if events.empty:
        return events
    events["lag_return_z"] = prior_z(events["report_to_release_return"], 104, 52)
    magnitude = events["position_surprise_z"].abs().clip(lower=0.5)
    events["assimilation_fraction"] = (
        np.sign(events["position_surprise_z"]) * events["lag_return_z"] / magnitude
    )
    return events


def fit_surprise_threshold(events: pd.DataFrame, quantile: float) -> float:
    start, end = WINDOWS["fit"]
    reference = events.loc[
        (events["release_time"] >= start) & (events["release_time"] < end),
        "position_surprise_z",
    ].abs().dropna()
    if len(reference) < 100:
        raise ValueError(f"insufficient fit reports: {len(reference)}")
    return float(reference.quantile(quantile))


def assimilation_signals(
    events: pd.DataFrame,
    rows: int,
    *,
    threshold: float,
    state: str,
    flip: bool = False,
    ignore_assimilation: bool = False,
    release_extra_weeks: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    surprise = events["position_surprise_z"].to_numpy(float)
    fraction = events["assimilation_fraction"].to_numpy(float)
    selected = np.isfinite(surprise) & np.isfinite(fraction) & (np.abs(surprise) >= threshold)
    if not ignore_assimilation:
        if state == "unpriced":
            selected &= fraction <= 0.0
        elif state == "over_assimilated":
            selected &= fraction >= 1.0
        else:
            raise KeyError(state)
    side = np.sign(surprise) * (1 if state == "unpriced" else -1)
    if flip:
        side = -side
    positions = events["signal_position"].to_numpy(int) + release_extra_weeks * 7 * 288
    selected &= positions < rows
    long_active = np.zeros(rows, dtype=bool)
    short_active = np.zeros(rows, dtype=bool)
    long_active[positions[selected & (side > 0.0)]] = True
    short_active[positions[selected & (side < 0.0)]] = True
    return long_active, short_active


def main() -> None:
    market, dates, reports = load_pre2024()
    holds = (576, 1152)
    extremes = {
        hold: (
            _future_extreme(market["low"].to_numpy(float), hold, "min"),
            _future_extreme(market["high"].to_numpy(float), hold, "max"),
        )
        for hold in holds
    }
    event_bank = {
        participant: build_assimilation_events(
            market,
            dates,
            reports,
            participant=participant,
        )
        for participant in PARTICIPANT_COLUMNS
    }
    rows: list[dict[str, Any]] = []
    signal_bank: dict[tuple[str, float, str], tuple[np.ndarray, np.ndarray]] = {}
    for participant, tail, state in itertools.product(
        PARTICIPANT_COLUMNS,
        (0.80, 0.90),
        ("unpriced", "over_assimilated"),
    ):
        events = event_bank[participant]
        threshold = fit_surprise_threshold(events, tail)
        long_active, short_active = assimilation_signals(
            events,
            len(market),
            threshold=threshold,
            state=state,
        )
        signal_bank[(participant, tail, state)] = (long_active, short_active)
        for hold in holds:
            stats = simulate(market, dates, long_active, short_active, hold, extremes[hold])
            rows.append(
                {
                    "participant": participant,
                    "tail": tail,
                    "state": state,
                    "hold": hold,
                    "threshold": threshold,
                    "signals": int((long_active | short_active).sum()),
                    "rank": rank_key(stats),
                    "prelim_admitted": admission(stats),
                    "stats": stats,
                }
            )
    rows.sort(key=lambda row: row["rank"], reverse=True)
    print(
        "reports",
        len(reports),
        "candidates",
        len(rows),
        "admitted",
        sum(row["prelim_admitted"] for row in rows),
    )
    for index, row in enumerate(rows, 1):
        print_stats(
            f"RANK {index} {row['participant']} q{row['tail']} {row['state']} "
            f"h{row['hold']} sig={row['signals']} rank={row['rank']}",
            row["stats"],
        )

    top = rows[0]
    events = event_bank[top["participant"]]
    signal_kwargs = {
        "threshold": top["threshold"],
        "state": top["state"],
    }
    long_active, short_active = signal_bank[
        (top["participant"], top["tail"], top["state"])
    ]
    hold = top["hold"]
    controls: dict[str, dict[str, dict[str, Any]]] = {}
    for name, extra in (
        ("direction_flip", {"flip": True}),
        ("ignore_assimilation", {"ignore_assimilation": True}),
        ("release_extra_4w", {"release_extra_weeks": 4}),
    ):
        control_long, control_short = assimilation_signals(
            events,
            len(market),
            **signal_kwargs,
            **extra,
        )
        controls[name] = simulate(
            market,
            dates,
            control_long,
            control_short,
            hold,
            extremes[hold],
        )
    other_participant = next(
        participant for participant in PARTICIPANT_COLUMNS if participant != top["participant"]
    )
    other_events = event_bank[other_participant]
    other_threshold = fit_surprise_threshold(other_events, top["tail"])
    control_long, control_short = assimilation_signals(
        other_events,
        len(market),
        threshold=other_threshold,
        state=top["state"],
    )
    controls["participant_swap"] = simulate(
        market,
        dates,
        control_long,
        control_short,
        hold,
        extremes[hold],
    )
    for name, stats in controls.items():
        print_stats("CONTROL " + name, stats)

    cost_stress = {
        str(side_bp): simulate(
            market,
            dates,
            long_active,
            short_active,
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
            "release": "CFTC report date +8d",
            "grid_size": 16,
            "surprise": "prior-only z-score of weekly participant net-share change",
            "assimilation": "prior-only z-score of report-to-release BTC return divided by signed surprise magnitude",
            "state_routes": {
                "unpriced": "fraction <=0, follow position surprise",
                "over_assimilated": "fraction >=1, fade position surprise",
            },
            "entry": "first completed 5m decision bar at/after conservative release; next 5m open",
            "leverage": 0.5,
            "cost": "6bp/side implementation cost",
            "strict_mdd": "favorable-first adverse-second OHLC high-water",
            "invalid_control_not_run": "report-date availability would be lookahead and is documented, not simulated",
            "oos_opened": False,
        },
        "rows": [{**row, "rank": list(row["rank"])} for row in rows],
        "controls": controls,
        "cost_stress": cost_stress,
        "final_admitted": bool(
            top["prelim_admitted"] and not any(admission(stats) for stats in controls.values())
        ),
    }
    Path("results/cftc_release_assimilation_alpha_scan_2026-07-13.json").write_text(
        json.dumps(result, indent=2)
    )


if __name__ == "__main__":
    main()
