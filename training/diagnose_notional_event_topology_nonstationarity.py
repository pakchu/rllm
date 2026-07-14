"""Post-hoc NETF v1 year/structure nonstationarity decomposition."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from training.diagnose_metaorder_fragmentation_impact_curvature_failure import (
    build_trade_ledger,
)
from training.evaluate_metaorder_fragmentation_impact_curvature import (
    EvaluationConfig,
    _sha256,
    simulate_schedule,
)
from training.evaluate_notional_event_topology_fracture import (
    _verify_preregistration,
)
from training.preregister_metaorder_fragmentation_impact_curvature import (
    Config as SourceConfig,
    load_causal_frame,
)
from training.preregister_notional_event_topology_fracture import (
    CANDIDATES,
    Config as SignalConfig,
    compute_netf,
    nonoverlapping_netf_schedule,
)


SELECTION_RESULT = Path(
    "results/notional_event_topology_fracture_selection_2026-07-14.json"
)
SELECTION_RESULT_SHA256 = (
    "478e6562ce59d6bfc93257d096381ccc19d5989a008771dfa9b797fb1334522a"
)
OUTPUT = Path(
    "results/notional_event_topology_fracture_nonstationarity_2026-07-14.json"
)
MARK_COLUMNS = (
    "arrival_burst_mark",
    "notional_concentration_mark",
    "trade_id_span_per_aggregate_event_mark",
)


def _verify_rejected_selection() -> dict[str, Any]:
    _verify_preregistration()
    if _sha256(SELECTION_RESULT) != SELECTION_RESULT_SHA256:
        raise ValueError("NETF selection result changed after rejection")
    result = json.loads(SELECTION_RESULT.read_text())
    if result.get("selection", {}).get("rejected") is not True:
        raise ValueError("NETF selection is not a frozen rejection")
    if result.get("protocol", {}).get("sealed_windows") != [
        "test2024",
        "eval2025",
        "ytd2026",
    ]:
        raise ValueError("NETF selection did not preserve sealed OOS windows")
    return result


def structure_combo(row: pd.Series) -> str:
    return "".join("1" if bool(row[column]) else "0" for column in MARK_COLUMNS)


def _year_stats(
    frame: pd.DataFrame,
    signal: pd.DataFrame,
    year: int,
    cfg: EvaluationConfig,
) -> dict[str, Any]:
    start = f"{year}-01-01"
    end = f"{year + 1}-01-01"
    schedule = nonoverlapping_netf_schedule(
        signal, frame, start=start, end=end
    )
    return simulate_schedule(
        frame,
        schedule,
        start=start,
        end=end,
        cfg=cfg,
    )


def run_diagnostic() -> dict[str, Any]:
    selection = _verify_rejected_selection()
    frame, source = load_causal_frame(SourceConfig())
    signal_cfg = SignalConfig()
    execution_cfg = replace(EvaluationConfig(), output=str(OUTPUT))
    candidate_results: list[dict[str, Any]] = []
    slow_structure: list[dict[str, Any]] = []
    slow_origin_medians: list[dict[str, Any]] = []

    for candidate in CANDIDATES:
        signal = compute_netf(frame, candidate, signal_cfg)
        yearly = {
            str(year): _year_stats(frame, signal, year, execution_cfg)
            for year in range(2020, 2024)
        }
        candidate_results.append(
            {"candidate": candidate.name, "yearly": yearly}
        )
        if candidate.name != "netf_slow":
            continue

        origin_rows: list[dict[str, Any]] = []
        ledger_rows: list[dict[str, Any]] = []
        for year in range(2020, 2024):
            schedule = nonoverlapping_netf_schedule(
                signal,
                frame,
                start=f"{year}-01-01",
                end=f"{year + 1}-01-01",
            )
            ledger = build_trade_ledger(frame, schedule, execution_cfg)
            for offset, trade in enumerate(schedule.itertuples(index=False)):
                signal_position = int(trade.signal_position)
                origin = int(signal.loc[signal_position, "origin_position"])
                origin_signal = signal.loc[origin]
                combo = structure_combo(origin_signal)
                ledger_rows.append(
                    {
                        "year": year,
                        "combo": combo,
                        "account_gross_return": float(
                            ledger.loc[offset, "account_gross_return"]
                        ),
                        "account_net_return": float(
                            ledger.loc[offset, "account_net_return"]
                        ),
                    }
                )
                origin_rows.append(
                    {
                        "year": year,
                        "arrival_burstiness": float(
                            frame.loc[origin, "interarrival_burstiness"]
                        ),
                        "event_notional_hhi": float(
                            frame.loc[origin, "event_notional_hhi"]
                        ),
                        "trade_id_span_per_aggregate_event": float(
                            frame.loc[origin, "underlying_trades_per_agg_event"]
                        ),
                        "relative_topology_tension": float(
                            signal.loc[origin, "topology_tension"]
                            / signal.loc[origin, "tension_baseline"]
                        ),
                    }
                )

        ledger_frame = pd.DataFrame(ledger_rows)
        for (year, combo), group in ledger_frame.groupby(
            ["year", "combo"], sort=True, observed=True
        ):
            slow_structure.append(
                {
                    "year": int(year),
                    "combo": str(combo),
                    "trade_count": int(len(group)),
                    "mean_account_gross_bps": float(
                        group["account_gross_return"].mean() * 10_000.0
                    ),
                    "mean_account_net_bps": float(
                        group["account_net_return"].mean() * 10_000.0
                    ),
                    "net_win_rate": float(
                        group["account_net_return"].gt(0.0).mean()
                    ),
                }
            )
        origin_frame = pd.DataFrame(origin_rows)
        for year, group in origin_frame.groupby("year", sort=True, observed=True):
            slow_origin_medians.append(
                {
                    "year": int(year),
                    **{
                        column: float(group[column].median())
                        for column in group.columns
                        if column != "year"
                    },
                }
            )

    return {
        "protocol": {
            "name": "NETF v1 post-hoc nonstationarity diagnostic",
            "selection_result_sha256": SELECTION_RESULT_SHA256,
            "opened_windows_only": ["2020", "2021", "2022", "2023"],
            "sealed_windows_still_unopened": [
                "test2024",
                "eval2025",
                "ytd2026",
            ],
            "may_repair_or_promote_netf": False,
            "structure_combo_order": list(MARK_COLUMNS),
        },
        "selection_verdict": selection["selection"],
        "source": source,
        "candidates": candidate_results,
        "netf_slow_structure_combinations": slow_structure,
        "netf_slow_origin_feature_medians": slow_origin_medians,
    }


def main() -> None:
    result = run_diagnostic()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    summary = {
        "yearly": [
            {
                "candidate": item["candidate"],
                "stats": {
                    year: {
                        key: stats[key]
                        for key in (
                            "absolute_return_pct",
                            "cagr_pct",
                            "strict_mdd_pct",
                            "cagr_to_strict_mdd",
                            "trade_count",
                        )
                    }
                    for year, stats in item["yearly"].items()
                },
            }
            for item in result["candidates"]
        ],
        "output": str(OUTPUT),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
