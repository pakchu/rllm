"""Freeze outcome-blind clocks for the minute packet topology battery."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SELECTION_END = "2024-01-01"
ROLLING_WINDOW = 8_640
ROLLING_MIN_PERIODS = 2_016
HOLDS = (24, 48, 96)
SWARM_GAP_QUANTILES = (0.10, 0.20)
SWARM_IMPACT_QUANTILES = (0.20, 0.35)
CHURN_TICKET_QUANTILES = (0.70, 0.80)
CHURN_FLOW_QUANTILES = (0.20, 0.35)


@dataclass(frozen=True)
class Candidate:
    family: str
    hold_bars: int
    primary_quantile: float
    secondary_quantile: float

    @property
    def name(self) -> str:
        return (
            f"{self.family}_p{int(round(self.primary_quantile * 100)):02d}_"
            f"s{int(round(self.secondary_quantile * 100)):02d}_h{self.hold_bars}"
        )


CANDIDATES = tuple(
    Candidate("um_swarm_absorption", hold, gap_q, impact_q)
    for gap_q in SWARM_GAP_QUANTILES
    for impact_q in SWARM_IMPACT_QUANTILES
    for hold in HOLDS
) + tuple(
    Candidate("cross_venue_churn_breakout", hold, ticket_q, flow_q)
    for ticket_q in CHURN_TICKET_QUANTILES
    for flow_q in CHURN_FLOW_QUANTILES
    for hold in HOLDS
)


@dataclass(frozen=True)
class Config:
    input_csv: str = (
        "/home/pakchu/rllm/data/binance_cross_venue_minute_dispersion_btc/"
        "BTCUSDT_cross_venue_minute_dispersion_5m_2020-01_2023-12.csv.gz"
    )
    output: str = "results/minute_packet_topology_support_2026-07-19.json"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def load_source(path: str | Path, *, cutoff: str = SELECTION_END) -> pd.DataFrame:
    frame = pd.read_csv(path, compression="infer", parse_dates=["date"])
    frame = frame.sort_values("date").reset_index(drop=True)
    if frame.empty:
        raise ValueError("minute packet source is empty")
    if frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing:
        raise ValueError("minute packet source timestamps are duplicate or unordered")
    if frame["date"].max() >= pd.Timestamp(cutoff):
        raise ValueError("support source is not physically truncated before 2024")
    expected = pd.date_range(frame["date"].iloc[0], frame["date"].iloc[-1], freq="5min")
    if not frame["date"].equals(pd.Series(expected, name="date")):
        raise ValueError("minute packet source is not a gapless five-minute grid")
    if not frame["feature_available_time_utc"].astype(str).equals(
        (frame["date"] + pd.Timedelta("5min")).astype(str)
    ):
        raise ValueError("feature availability violates the completed-bar contract")
    return frame


def rolling_prior_quantile(
    values: pd.Series,
    valid: pd.Series,
    quantile: float,
    *,
    window: int = ROLLING_WINDOW,
    min_periods: int = ROLLING_MIN_PERIODS,
) -> pd.Series:
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be strictly between zero and one")
    return values.where(valid).shift(1).rolling(window, min_periods=min_periods).quantile(quantile)


def build_thresholds(frame: pd.DataFrame) -> dict[str, pd.Series]:
    valid = frame["minute_dispersion_feature_valid"].astype(bool)
    specs: dict[str, tuple[pd.Series, Iterable[float]]] = {
        "um_gap": (
            frame["um_quote_minus_trade_time_hhi"],
            SWARM_GAP_QUANTILES,
        ),
        "um_impact_efficiency": (
            frame["um_impact_per_abs_flow_fraction_bp"],
            SWARM_IMPACT_QUANTILES,
        ),
        "um_persistence": (frame["um_flow_sign_persistence"], (0.70,)),
        "um_ticket_dispersion": (
            frame["um_ticket_log_std"],
            CHURN_TICKET_QUANTILES,
        ),
        "um_abs_net_flow": (
            frame["um_net_flow_fraction"].abs(),
            CHURN_FLOW_QUANTILES,
        ),
        "um_abs_signed_impact": (frame["um_signed_impact_bp"].abs(), (0.80,)),
    }
    output: dict[str, pd.Series] = {}
    for name, (values, quantiles) in specs.items():
        for quantile in quantiles:
            output[f"{name}_q{int(round(quantile * 100)):02d}"] = rolling_prior_quantile(
                values, valid, quantile
            )
    return output


def _qkey(name: str, quantile: float) -> str:
    return f"{name}_q{int(round(quantile * 100)):02d}"


def candidate_clock(
    frame: pd.DataFrame,
    thresholds: dict[str, pd.Series],
    candidate: Candidate,
) -> tuple[pd.Series, pd.Series]:
    valid = frame["minute_dispersion_feature_valid"].astype(bool)
    um_flow = frame["um_net_flow_fraction"]
    spot_flow = frame["spot_net_flow_fraction"]
    if candidate.family == "um_swarm_absorption":
        active = (
            valid
            & frame["um_quote_minus_trade_time_hhi"].le(
                thresholds[_qkey("um_gap", candidate.primary_quantile)]
            )
            & frame["um_flow_sign_persistence"].ge(thresholds["um_persistence_q70"])
            & frame["um_flow_sign_switch_rate"].le(0.25)
            & frame["um_impact_per_abs_flow_fraction_bp"].le(
                thresholds[_qkey("um_impact_efficiency", candidate.secondary_quantile)]
            )
            & frame["net_flow_sign_agreement"].lt(0.0)
        )
        side = -np.sign(um_flow.fillna(0.0)).astype(np.int8)
    elif candidate.family == "cross_venue_churn_breakout":
        um_return_side = np.sign(
            frame["um_signed_impact_bp"].fillna(0.0) * um_flow.fillna(0.0)
        ).astype(np.int8)
        spot_return_side = np.sign(
            frame["spot_signed_impact_bp"].fillna(0.0) * spot_flow.fillna(0.0)
        ).astype(np.int8)
        active = (
            valid
            & frame["um_flow_sign_switch_rate"].ge(0.75)
            & frame["um_ticket_log_std"].ge(
                thresholds[_qkey("um_ticket_dispersion", candidate.primary_quantile)]
            )
            & um_flow.abs().le(
                thresholds[_qkey("um_abs_net_flow", candidate.secondary_quantile)]
            )
            & frame["um_signed_impact_bp"].abs().ge(thresholds["um_abs_signed_impact_q80"])
            & um_return_side.eq(spot_return_side)
        )
        side = um_return_side
    else:
        raise KeyError(candidate.family)
    active = active.fillna(False) & side.isin((-1, 1))
    onset = active & ~active.shift(1, fill_value=False)
    return onset.astype(bool), side.astype(np.int8)


def nonoverlapping_schedule(
    frame: pd.DataFrame,
    onset: pd.Series,
    side: pd.Series,
    *,
    hold_bars: int,
) -> pd.DataFrame:
    if hold_bars <= 0:
        raise ValueError("hold_bars must be positive")
    rows: list[dict[str, Any]] = []
    next_allowed = 0
    for signal_position in np.flatnonzero(onset.to_numpy(bool)):
        entry_position = int(signal_position) + 1
        exit_position = entry_position + int(hold_bars)
        if signal_position < next_allowed or exit_position >= len(frame):
            continue
        action = int(side.iloc[signal_position])
        if action not in (-1, 1):
            continue
        rows.append(
            {
                "signal_position": int(signal_position),
                "entry_position": entry_position,
                "exit_position": exit_position,
                "side": action,
                "signal_date": str(frame["date"].iloc[signal_position]),
                "entry_date": str(frame["date"].iloc[entry_position]),
                "exit_date": str(frame["date"].iloc[exit_position]),
            }
        )
        next_allowed = exit_position
    return pd.DataFrame(rows)


def support_summary(schedule: pd.DataFrame) -> dict[str, Any]:
    if schedule.empty:
        return {
            "total": 0,
            "longs": 0,
            "shorts": 0,
            "by_year": {},
            "by_2023_half": {"h1": 0, "h2": 0},
            "max_month_fraction": 1.0,
        }
    entry = pd.to_datetime(schedule["entry_date"])
    by_year = entry.dt.year.value_counts().sort_index()
    months = entry.dt.to_period("M").value_counts()
    total = len(schedule)
    return {
        "total": int(total),
        "longs": int((schedule["side"] > 0).sum()),
        "shorts": int((schedule["side"] < 0).sum()),
        "by_year": {str(int(key)): int(value) for key, value in by_year.items()},
        "by_2023_half": {
            "h1": int(((entry >= pd.Timestamp("2023-01-01")) & (entry < pd.Timestamp("2023-07-01"))).sum()),
            "h2": int(((entry >= pd.Timestamp("2023-07-01")) & (entry < pd.Timestamp("2024-01-01"))).sum()),
        },
        "max_month_fraction": float(months.max() / total),
    }


def support_gates(summary: dict[str, Any]) -> dict[str, bool]:
    total = int(summary["total"])
    sides = (int(summary["longs"]), int(summary["shorts"]))
    years = summary["by_year"]
    halves = summary["by_2023_half"]
    return {
        "total_at_least_150": total >= 150,
        "each_year_at_least_30": all(int(years.get(str(year), 0)) >= 30 for year in range(2020, 2024)),
        "each_2023_half_at_least_15": min(int(halves["h1"]), int(halves["h2"])) >= 15,
        "each_side_at_least_25pct": total > 0 and min(sides) / total >= 0.25,
        "max_month_fraction_at_most_15pct": float(summary["max_month_fraction"]) <= 0.15,
    }


def build_report(cfg: Config) -> dict[str, Any]:
    frame = load_source(cfg.input_csv)
    thresholds = build_thresholds(frame)
    candidates: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        onset, side = candidate_clock(frame, thresholds, candidate)
        schedule = nonoverlapping_schedule(
            frame, onset, side, hold_bars=candidate.hold_bars
        )
        summary = support_summary(schedule)
        gates = support_gates(summary)
        candidates.append(
            {
                "candidate": asdict(candidate),
                "name": candidate.name,
                "clock_hash": canonical_hash(schedule.to_dict(orient="records")),
                "support": summary,
                "gates": gates,
                "passes_support": bool(all(gates.values())),
            }
        )
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "outcomes_opened": False,
            "selection_end_exclusive": SELECTION_END,
            "signal_available_after_completed_bar": True,
            "entry": "next five-minute open",
            "exit": "fixed future five-minute open",
            "thresholds": "strictly prior rolling quantiles; current row excluded",
            "rolling_window_bars": ROLLING_WINDOW,
            "rolling_min_periods": ROLLING_MIN_PERIODS,
            "source_interpretation": "five one-minute aggregate topology, not trade-level HHI",
        },
        "config": asdict(cfg),
        "source": {
            "path": cfg.input_csv,
            "sha256": sha256_file(cfg.input_csv),
            "rows": int(len(frame)),
            "valid_rows": int(frame["minute_dispersion_feature_valid"].sum()),
            "first_date": str(frame["date"].min()),
            "last_date": str(frame["date"].max()),
        },
        "candidate_count": len(candidates),
        "support_pass_count": int(sum(item["passes_support"] for item in candidates)),
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", default=Config.input_csv)
    parser.add_argument("--output", default=Config.output)
    cfg = Config(**vars(parser.parse_args()))
    report = build_report(cfg)
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "candidate_count": report["candidate_count"],
                "support_pass_count": report["support_pass_count"],
                "source_sha256": report["source"]["sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
