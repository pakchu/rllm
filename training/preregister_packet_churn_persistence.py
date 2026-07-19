"""Freeze the outcome-blind packet-churn persistence candidate clock.

This successor observes six completed five-minute bars after a cross-venue
packet-churn setup.  It does not load an execution price, funding payment, or
post-entry return.  The resulting clock is therefore a support artifact, not a
profitability result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from training.preregister_minute_packet_topology_alpha import (
    Candidate,
    candidate_clock,
    load_source,
    rolling_prior_quantile,
)


SOURCE_SHA256 = "5ea9f5075171c255732cc6eed003736c1beed211a0e6fd7797ab02f31a917aaa"
CONFIRMATION_BARS = 6
HOLD_BARS = 96
ENTRY_DELAY_BARS = 2
MIN_CONFIRMATION_CHURN_BARS = 3
PRIMARY_QUANTILES = (0.70, 0.80)
SECONDARY_QUANTILES = (0.20, 0.35)


@dataclass(frozen=True)
class Config:
    input_csv: str = (
        "/home/pakchu/rllm/data/binance_cross_venue_minute_dispersion_btc/"
        "BTCUSDT_cross_venue_minute_dispersion_5m_2020-01_2023-12.csv.gz"
    )
    output: str = "results/packet_churn_persistence_support_2026-07-19.json"
    clock_output: str = "results/packet_churn_persistence_clock_2026-07-19.csv"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def _raw_return_bp(frame: pd.DataFrame, venue: str) -> pd.Series:
    """Recover completed-bar raw return from flow-signed impact.

    The frozen source stores ``signed_impact_bp = sign(net flow) * return_bp``.
    Multiplying once more by the flow sign recovers the contemporaneously
    completed raw return.  A zero-flow row is unavailable for confirmation.
    """

    flow = frame[f"{venue}_net_flow_fraction"]
    signed_impact = frame[f"{venue}_signed_impact_bp"]
    flow_sign = flow.map(np.sign).replace(0.0, np.nan)
    return signed_impact * flow_sign


def build_pcp_thresholds(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Build only the causal prior thresholds consumed by the PCP clock."""

    valid = cast(pd.Series, frame["minute_dispersion_feature_valid"]).astype(bool)
    ticket_dispersion = cast(pd.Series, frame["um_ticket_log_std"])
    absolute_net_flow = cast(pd.Series, frame["um_net_flow_fraction"]).abs()
    absolute_signed_impact = cast(pd.Series, frame["um_signed_impact_bp"]).abs()
    output = {
        f"um_ticket_dispersion_q{int(round(quantile * 100)):02d}": (
            rolling_prior_quantile(ticket_dispersion, valid, quantile)
        )
        for quantile in PRIMARY_QUANTILES
    }
    output.update(
        {
            f"um_abs_net_flow_q{int(round(quantile * 100)):02d}": (
                rolling_prior_quantile(absolute_net_flow, valid, quantile)
            )
            for quantile in SECONDARY_QUANTILES
        }
    )
    output["um_abs_signed_impact_q80"] = rolling_prior_quantile(
        absolute_signed_impact, valid, 0.80
    )
    return output


def build_schedule(
    frame: pd.DataFrame,
    thresholds: dict[str, pd.Series],
    candidate: Candidate,
) -> tuple[pd.DataFrame, int]:
    """Build a fixed non-overlapping schedule from pre-entry observations."""

    onset, side = candidate_clock(frame, thresholds, candidate)
    valid = frame["minute_dispersion_feature_valid"].astype(bool)
    um_return_bp = _raw_return_bp(frame, "um")
    spot_return_bp = _raw_return_bp(frame, "spot")
    rows: list[dict[str, Any]] = []
    raw_qualified = 0
    next_allowed_setup = 0

    for setup_position in np.flatnonzero(onset.to_numpy(bool)):
        confirmation_end_position = int(setup_position) + CONFIRMATION_BARS
        entry_position = confirmation_end_position + ENTRY_DELAY_BARS
        exit_position = entry_position + HOLD_BARS
        if exit_position >= len(frame):
            continue

        confirmation = slice(int(setup_position) + 1, confirmation_end_position + 1)
        if not bool(valid.iloc[confirmation].all()):
            continue
        action = int(side.iloc[setup_position])
        if action not in (-1, 1):
            continue

        um_confirmation = um_return_bp.iloc[confirmation]
        spot_confirmation = spot_return_bp.iloc[confirmation]
        if not bool(um_confirmation.notna().all() and spot_confirmation.notna().all()):
            continue
        um_displacement_bp = action * float(um_confirmation.sum())
        spot_displacement_bp = action * float(spot_confirmation.sum())
        churn_bars = int(
            frame["um_flow_sign_switch_rate"].iloc[confirmation].ge(0.50).sum()
        )
        if not (
            np.isfinite(um_displacement_bp)
            and np.isfinite(spot_displacement_bp)
            and um_displacement_bp > 0.0
            and spot_displacement_bp > 0.0
            and churn_bars >= MIN_CONFIRMATION_CHURN_BARS
        ):
            continue

        raw_qualified += 1
        if int(setup_position) < next_allowed_setup:
            continue
        rows.append(
            {
                "setup_position": int(setup_position),
                "confirmation_end_position": confirmation_end_position,
                "entry_position": entry_position,
                "exit_position": exit_position,
                "side": action,
                "setup_bar_date": str(frame["date"].iloc[setup_position]),
                "confirmation_end_bar_date": str(
                    frame["date"].iloc[confirmation_end_position]
                ),
                "signal_available_at": str(
                    frame["date"].iloc[confirmation_end_position] + pd.Timedelta("5min")
                ),
                "entry_date": str(frame["date"].iloc[entry_position]),
                "exit_date": str(frame["date"].iloc[exit_position]),
                "um_confirmation_displacement_bp": um_displacement_bp,
                "spot_confirmation_displacement_bp": spot_displacement_bp,
                "confirmation_churn_bars": churn_bars,
            }
        )
        next_allowed_setup = exit_position

    columns = [
        "setup_position",
        "confirmation_end_position",
        "entry_position",
        "exit_position",
        "side",
        "setup_bar_date",
        "confirmation_end_bar_date",
        "signal_available_at",
        "entry_date",
        "exit_date",
        "um_confirmation_displacement_bp",
        "spot_confirmation_displacement_bp",
        "confirmation_churn_bars",
    ]
    return pd.DataFrame.from_records(rows).reindex(columns=columns), raw_qualified


def support_summary(schedule: pd.DataFrame) -> dict[str, Any]:
    if schedule.empty:
        return {
            "total": 0,
            "longs": 0,
            "shorts": 0,
            "by_year": {},
            "train_2020_2022": {
                "total": 0,
                "longs": 0,
                "shorts": 0,
                "max_month_fraction": 1.0,
            },
            "selection_2023": {
                "total": 0,
                "longs": 0,
                "shorts": 0,
                "h1": 0,
                "h2": 0,
                "max_month_fraction": 1.0,
            },
        }
    entries = pd.to_datetime(schedule["entry_date"])
    total = len(schedule)
    by_year = entries.dt.year.value_counts().sort_index()

    def period_summary(start: str, end: str) -> dict[str, Any]:
        mask = (entries >= pd.Timestamp(start)) & (entries < pd.Timestamp(end))
        subset = schedule.loc[mask]
        subset_entries = entries.loc[mask]
        subset_total = len(subset)
        months = subset_entries.dt.to_period("M").value_counts()
        return {
            "total": int(subset_total),
            "longs": int(subset["side"].gt(0).sum()),
            "shorts": int(subset["side"].lt(0).sum()),
            "max_month_fraction": float(months.max() / subset_total)
            if subset_total
            else 1.0,
        }

    train = period_summary("2020-01-01", "2023-01-01")
    selection = period_summary("2023-01-01", "2024-01-01")
    selection["h1"] = int(
        (
            (entries >= pd.Timestamp("2023-01-01"))
            & (entries < pd.Timestamp("2023-07-01"))
        ).sum()
    )
    selection["h2"] = int(
        (
            (entries >= pd.Timestamp("2023-07-01"))
            & (entries < pd.Timestamp("2024-01-01"))
        ).sum()
    )
    return {
        "total": int(total),
        "longs": int(schedule["side"].gt(0).sum()),
        "shorts": int(schedule["side"].lt(0).sum()),
        "by_year": {str(int(year)): int(count) for year, count in by_year.items()},
        "train_2020_2022": train,
        "selection_2023": selection,
    }


def support_gates(summary: dict[str, Any]) -> dict[str, bool]:
    train = summary["train_2020_2022"]
    total = int(train["total"])
    return {
        "train_total_at_least_130": total >= 130,
        "each_train_year_at_least_40": all(
            int(summary["by_year"].get(str(year), 0)) >= 40
            for year in range(2020, 2023)
        ),
        "each_train_side_at_least_25pct": total > 0
        and min(int(train["longs"]), int(train["shorts"])) / total >= 0.25,
        "train_max_month_fraction_at_most_15pct": float(train["max_month_fraction"])
        <= 0.15,
    }


def build_report(cfg: Config) -> tuple[dict[str, Any], pd.DataFrame]:
    source_hash = sha256_file(cfg.input_csv)
    if source_hash != SOURCE_SHA256:
        raise ValueError("packet-churn source changed from the frozen predecessor")
    frame = load_source(cfg.input_csv)
    thresholds = build_pcp_thresholds(frame)

    trials: list[dict[str, Any]] = []
    schedules: dict[str, pd.DataFrame] = {}
    for primary in PRIMARY_QUANTILES:
        for secondary in SECONDARY_QUANTILES:
            candidate = Candidate(
                family="cross_venue_churn_breakout",
                hold_bars=HOLD_BARS,
                primary_quantile=primary,
                secondary_quantile=secondary,
            )
            schedule, raw_qualified = build_schedule(frame, thresholds, candidate)
            summary = support_summary(schedule)
            gates = support_gates(summary)
            name = f"pcp_{candidate.name}_confirm{CONFIRMATION_BARS}"
            schedules[name] = schedule
            trials.append(
                {
                    "name": name,
                    "base_candidate": asdict(candidate),
                    "raw_qualified_sequences": raw_qualified,
                    "support": summary,
                    "gates": gates,
                    "passes_support": all(gates.values()),
                    "clock_hash": canonical_hash(schedule.to_dict(orient="records")),
                }
            )

    passing = [trial for trial in trials if trial["passes_support"]]
    if len(passing) != 1:
        raise ValueError(
            f"expected exactly one support-passing cell, found {len(passing)}"
        )
    selected = passing[0]
    selected_clock = schedules[str(selected["name"])]
    report: dict[str, Any] = {
        "protocol": {
            "name": "PCP-6 packet churn persistence support freeze",
            "outcomes_opened": False,
            "post_entry_prices_loaded": False,
            "funding_loaded": False,
            "predecessor_train_outcomes_known": True,
            "successor_train_outcomes_opened": False,
            "successor_2023_outcomes_opened": False,
            "support_selection_uses_2023_incidence": False,
            "2023_incidence_disclosed_but_profitability_unopened": True,
            "sealed_windows": [
                "train_2020_2022",
                "selection_2023",
                "test_2024",
                "eval_2025",
                "holdout_2026",
            ],
            "threshold_or_direction_repair_after_returns_allowed": False,
        },
        "mechanism": {
            "base_event": "cross-venue one-minute packet churn breakout onset",
            "confirmation_bars": CONFIRMATION_BARS,
            "confirmation": "both USD-M and Spot cumulative completed-bar displacement remain in setup direction and at least three USD-M bars retain >=0.50 minute-flow sign-switch rate",
            "signal_clock": "sixth confirmation bar close",
            "entry_delay_bars_from_confirmation_bar": ENTRY_DELAY_BARS,
            "entry": "after one complete five-minute latency bar, at the following open",
            "hold_bars": HOLD_BARS,
            "action": "follow the original completed-bar displacement direction",
            "candidate_clock_released_by_failed_confirmation": True,
            "candidate_clock_released_by_overlap": False,
        },
        "support_stopping_rule": {
            "grid": {
                "ticket_dispersion_quantiles": list(PRIMARY_QUANTILES),
                "absolute_net_flow_quantiles": list(SECONDARY_QUANTILES),
            },
            "selection": "the unique cell passing every 2020-2022 outcome-blind count, time-dispersion, and side-balance gate; 2023 incidence is reported but does not select the cell",
            "trials": trials,
            "selected_name": selected["name"],
        },
        "source": {
            "path": cfg.input_csv,
            "sha256": source_hash,
            "rows": int(len(frame)),
            "valid_rows": int(
                frame["minute_dispersion_feature_valid"].astype(bool).sum()
            ),
            "first_date": str(frame["date"].iloc[0]),
            "last_date": str(frame["date"].iloc[-1]),
            "physical_end_exclusive": "2024-01-01 00:00:00",
            "columns_used": [
                "date",
                "feature_available_time_utc",
                "minute_dispersion_feature_valid",
                "um_net_flow_fraction",
                "spot_net_flow_fraction",
                "um_flow_sign_switch_rate",
                "um_ticket_log_std",
                "um_signed_impact_bp",
                "spot_signed_impact_bp",
            ],
        },
        "implementation": {
            "preregistration_source": "training/preregister_packet_churn_persistence.py",
            "preregistration_source_sha256": sha256_file(
                "training/preregister_packet_churn_persistence.py"
            ),
            "predecessor_source": "training/preregister_minute_packet_topology_alpha.py",
            "predecessor_source_sha256": sha256_file(
                "training/preregister_minute_packet_topology_alpha.py"
            ),
        },
        "selected": selected,
        "clock": {
            "path": cfg.clock_output,
            "rows": int(len(selected_clock)),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    report["result_hash"] = canonical_hash(
        {k: v for k, v in report.items() if k != "created_at"}
    )
    return report, selected_clock


def write_artifacts(cfg: Config, report: dict[str, Any], clock: pd.DataFrame) -> None:
    output = Path(cfg.output)
    clock_output = Path(cfg.clock_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    clock_output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or clock_output.exists():
        raise FileExistsError("PCP support artifacts are write-once")
    clock.to_csv(clock_output, index=False, lineterminator="\n")
    report["clock"]["sha256"] = sha256_file(clock_output)
    report["result_hash"] = canonical_hash(
        {k: v for k, v in report.items() if k not in {"created_at", "result_hash"}}
    )
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", default=Config.input_csv)
    parser.add_argument("--output", default=Config.output)
    parser.add_argument("--clock-output", default=Config.clock_output)
    args = parser.parse_args()
    cfg = Config(args.input_csv, args.output, args.clock_output)
    report, clock = build_report(cfg)
    write_artifacts(cfg, report, clock)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
