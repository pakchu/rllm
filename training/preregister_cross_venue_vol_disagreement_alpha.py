"""Freeze outcome-blind clocks for the BTC BVOL/DVOL disagreement battery."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SELECTION_END = "2024-01-01"
ROLLING_WINDOW_HOURS = 720
ROLLING_MIN_HOURS = 168
VOL_TAIL_QUANTILES = (0.80, 0.90)
PRICE_TAIL_QUANTILES = (0.80, 0.90)
HOLD_HOURS = (12, 24, 48)


@dataclass(frozen=True)
class Candidate:
    family: str
    vol_tail_quantile: float
    price_tail_quantile: float
    hold_hours: int

    @property
    def name(self) -> str:
        return (
            f"{self.family}_v{int(round(self.vol_tail_quantile * 100)):02d}_"
            f"p{int(round(self.price_tail_quantile * 100)):02d}_h{self.hold_hours}"
        )


CANDIDATES = tuple(
    Candidate(family, vol_q, price_q, hold)
    for family in ("bvol_rich_move_fade", "dvol_rich_move_follow")
    for vol_q in VOL_TAIL_QUANTILES
    for price_q in PRICE_TAIL_QUANTILES
    for hold in HOLD_HOURS
)


@dataclass(frozen=True)
class Config:
    input_csv: str = (
        "/home/pakchu/rllm/data/cross_venue_vol_disagreement_btc/"
        "BTC_cross_venue_vol_disagreement_1h_pre2024.csv.gz"
    )
    output: str = "results/cross_venue_vol_disagreement_support_2026-07-19.json"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def load_source(path: str | Path) -> pd.DataFrame:
    date_columns = [
        "signal_time_utc",
        "feature_available_time_utc",
        "trade_earliest_time_utc",
    ]
    frame = pd.read_csv(path, compression="infer", parse_dates=date_columns)
    frame = frame.sort_values("signal_time_utc").reset_index(drop=True)
    if frame.empty or frame["signal_time_utc"].duplicated().any():
        raise ValueError("volatility disagreement source is empty or duplicated")
    expected = pd.date_range(
        frame["signal_time_utc"].iloc[0], frame["signal_time_utc"].iloc[-1], freq="1h"
    )
    if not frame["signal_time_utc"].equals(pd.Series(expected, name="signal_time_utc")):
        raise ValueError("volatility disagreement source is not an hourly grid")
    if frame["signal_time_utc"].max() >= pd.Timestamp(SELECTION_END):
        raise ValueError("support source is not physically truncated before 2024")
    if not frame["feature_available_time_utc"].equals(frame["signal_time_utc"]):
        raise ValueError("feature availability differs from completed-hour signal time")
    if not frame["trade_earliest_time_utc"].equals(
        frame["signal_time_utc"] + pd.Timedelta("5min")
    ):
        raise ValueError("source does not enforce the five-minute execution delay")
    return frame


def prior_quantile(
    values: pd.Series,
    valid: pd.Series,
    quantile: float,
    *,
    window: int = ROLLING_WINDOW_HOURS,
    min_periods: int = ROLLING_MIN_HOURS,
) -> pd.Series:
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be strictly between zero and one")
    return values.where(valid).shift(1).rolling(window, min_periods=min_periods).quantile(quantile)


def build_thresholds(frame: pd.DataFrame) -> dict[str, pd.Series]:
    valid = frame["feature_valid"].astype(bool)
    output: dict[str, pd.Series] = {}
    for quantile in VOL_TAIL_QUANTILES:
        suffix = int(round(quantile * 100))
        output[f"ratio_high_q{suffix:02d}"] = prior_quantile(
            frame["log_bvol_dvol_ratio"], valid, quantile
        )
        output[f"ratio_low_q{suffix:02d}"] = prior_quantile(
            frame["log_bvol_dvol_ratio"], valid, 1.0 - quantile
        )
    for quantile in PRICE_TAIL_QUANTILES:
        suffix = int(round(quantile * 100))
        output[f"abs_move_q{suffix:02d}"] = prior_quantile(
            frame["btc_return_4h"].abs(), valid, quantile
        )
    return output


def candidate_clock(
    frame: pd.DataFrame,
    thresholds: dict[str, pd.Series],
    candidate: Candidate,
) -> tuple[pd.Series, pd.Series]:
    valid = frame["feature_valid"].astype(bool)
    ratio = frame["log_bvol_dvol_ratio"]
    move = frame["btc_return_4h"]
    vol_suffix = int(round(candidate.vol_tail_quantile * 100))
    price_suffix = int(round(candidate.price_tail_quantile * 100))
    large_move = move.abs().ge(thresholds[f"abs_move_q{price_suffix:02d}"])
    if candidate.family == "bvol_rich_move_fade":
        active = valid & ratio.ge(thresholds[f"ratio_high_q{vol_suffix:02d}"]) & large_move
        side = -np.sign(move.fillna(0.0)).astype(np.int8)
    elif candidate.family == "dvol_rich_move_follow":
        active = valid & ratio.le(thresholds[f"ratio_low_q{vol_suffix:02d}"]) & large_move
        side = np.sign(move.fillna(0.0)).astype(np.int8)
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
    hold_hours: int,
) -> pd.DataFrame:
    if hold_hours <= 0:
        raise ValueError("hold_hours must be positive")
    rows: list[dict[str, Any]] = []
    next_allowed = pd.Timestamp.min
    cutoff = pd.Timestamp(SELECTION_END)
    for position in np.flatnonzero(onset.to_numpy(bool)):
        signal_time = frame["signal_time_utc"].iloc[position]
        entry_time = frame["trade_earliest_time_utc"].iloc[position]
        exit_time = entry_time + pd.Timedelta(hours=hold_hours)
        action = int(side.iloc[position])
        if entry_time < next_allowed or exit_time >= cutoff or action not in (-1, 1):
            continue
        rows.append(
            {
                "signal_time": str(signal_time),
                "entry_time": str(entry_time),
                "exit_time": str(exit_time),
                "side": action,
            }
        )
        next_allowed = exit_time
    return pd.DataFrame(rows, columns=["signal_time", "entry_time", "exit_time", "side"])


def support_summary(schedule: pd.DataFrame) -> dict[str, Any]:
    if schedule.empty:
        return {
            "total": 0,
            "longs": 0,
            "shorts": 0,
            "q3": 0,
            "q4": 0,
            "by_month": {},
            "max_month_fraction": 1.0,
        }
    entry = pd.to_datetime(schedule["entry_time"])
    months = entry.dt.to_period("M").value_counts().sort_index()
    total = len(schedule)
    return {
        "total": int(total),
        "longs": int((schedule["side"] > 0).sum()),
        "shorts": int((schedule["side"] < 0).sum()),
        "q3": int(((entry >= "2023-07-01") & (entry < "2023-10-01")).sum()),
        "q4": int(((entry >= "2023-10-01") & (entry < "2024-01-01")).sum()),
        "by_month": {str(key): int(value) for key, value in months.items()},
        "max_month_fraction": float(months.max() / total),
    }


def support_gates(summary: dict[str, Any]) -> dict[str, bool]:
    total = int(summary["total"])
    return {
        "total_at_least_24": total >= 24,
        "each_quarter_at_least_8": min(int(summary["q3"]), int(summary["q4"])) >= 8,
        "each_side_at_least_25pct": total > 0
        and min(int(summary["longs"]), int(summary["shorts"])) / total >= 0.25,
        "max_month_fraction_at_most_35pct": float(summary["max_month_fraction"]) <= 0.35,
    }


def build_report(cfg: Config) -> dict[str, Any]:
    frame = load_source(cfg.input_csv)
    thresholds = build_thresholds(frame)
    candidates: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        onset, side = candidate_clock(frame, thresholds, candidate)
        schedule = nonoverlapping_schedule(
            frame, onset, side, hold_hours=candidate.hold_hours
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
    stable = {
        "protocol": {
            "outcomes_opened": False,
            "selection_end_exclusive": SELECTION_END,
            "thresholds": "720h strictly-prior rolling quantiles; current hour excluded",
            "rolling_min_hours": ROLLING_MIN_HOURS,
            "entry": "five minutes after completed hourly features",
            "exit": "fixed elapsed-hour hold; global non-overlap per candidate",
            "candidate_count": len(CANDIDATES),
            "direction_repair_allowed": False,
            "parameter_repair_after_returns_allowed": False,
        },
        "source": {"path": cfg.input_csv, "sha256": sha256_file(cfg.input_csv)},
        "candidates": candidates,
    }
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **stable,
        "support_freeze_hash": canonical_hash(stable),
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
                "candidates": len(report["candidates"]),
                "support_passes": sum(item["passes_support"] for item in report["candidates"]),
                "support_freeze_hash": report["support_freeze_hash"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
