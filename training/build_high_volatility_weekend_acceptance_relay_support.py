"""Build source-only HVWAR-24 clocks before Gross9 or economic outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import preregister_high_volatility_weekend_acceptance_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
from training.build_scheduled_trend_concordance_relay_support import load_market


PREREG_SHA = "0130c8492de4db2e0251993d937eee22d552c57f820a9c7701fa82b1384cefd0"
HELPER = Path("training/build_scheduled_trend_concordance_relay_support.py")
HELPER_SHA = "8ca554d88506df277434f73e5eb8850426614a880110088eb91aaae3b23c154f"
END = pd.Timestamp("2026-08-01T00:00:00Z")
STATE = Path("data/high_volatility_weekend_acceptance_relay_sources_2023_2026/weekend_states.csv.gz")
CLOCK = Path("data/high_volatility_weekend_acceptance_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_weekend_acceptance_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_weekend_acceptance_relay_support_2026-08-09.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), END),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = ("no_range_gate", "no_acceptance_gate", "central_close_rejection", "sunday_only_geometry", "direction_flip", "same_clock_forced_long")
COLUMNS = ("candidate", "control", "split", "decision_time", "feature_available_time", "entry_time", "exit_time", "side", "displacement", "range_log", "close_location", "range_rank")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geometry(window: pd.DataFrame) -> tuple[float, float, float] | None:
    values = window[["open", "high", "low", "close"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not (values > 0).all():
        return None
    if not (window.high.ge(window[["open", "close"]].max(axis=1)).all() and window.low.le(window[["open", "close"]].min(axis=1)).all() and window.high.ge(window.low).all()):
        return None
    low, high = float(window.low.min()), float(window.high.max())
    if high <= low:
        return None
    displacement = float(np.log(float(window.close.iloc[-1]) / float(window.open.iloc[0])))
    return displacement, float(np.log(high / low)), float((float(window.close.iloc[-1]) - low) / (high - low))


def score_states(market: pd.DataFrame, hours: int = 48) -> pd.DataFrame:
    if hours not in (24, 48):
        raise ValueError(hours)
    frame = market.copy()
    frame["date"] = pd.to_datetime(frame.date, utc=True)
    frame = frame.sort_values("date").set_index("date")
    decisions = frame.index[(frame.index.weekday == 0) & (frame.index.hour == 0) & (frame.index.minute == 0)]
    rows, prior = [], []
    bars = hours * 12
    for decision in decisions:
        expected = pd.date_range(decision - pd.Timedelta(hours=hours), decision, freq="5min", inclusive="left")
        window = frame.reindex(expected)
        value = geometry(window) if len(window) == bars else None
        if value is None:
            continue
        displacement, range_log, close_location = value
        history = np.asarray(prior[-90:], dtype=float)
        rank = float(((history < range_log).sum() + .5 * (history == range_log).sum()) / len(history)) if len(history) >= 60 else np.nan
        rows.append({"decision_time": decision, "displacement": displacement, "range_log": range_log, "close_location": close_location, "range_rank": rank})
        prior.append(range_log)
    return pd.DataFrame(rows)


def _accepted(row: pd.Series) -> bool:
    return bool((row.displacement > 0 and row.close_location >= .75) or (row.displacement < 0 and row.close_location <= .25))


def build_clock(full: pd.DataFrame, sunday: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    source = sunday if control == "sunday_only_geometry" else full
    rows = []
    for _, row in source.iterrows():
        ranked = bool(np.isfinite(row.range_rank) and row.range_rank >= .65)
        accepted = _accepted(row)
        side = int(np.sign(row.displacement)) if row.displacement != 0 else 0
        eligible = ranked and accepted and side != 0
        if control == "no_range_gate":
            eligible = accepted and side != 0
        elif control == "no_acceptance_gate":
            eligible = ranked and side != 0
        elif control == "central_close_rejection":
            eligible = ranked and .40 <= row.close_location <= .60 and side != 0
            side = -side
        if not eligible:
            continue
        if control == "direction_flip":
            side = -side
        elif control == "same_clock_forced_long":
            side = 1
        decision = pd.Timestamp(row.decision_time)
        entry, exit_ = decision + pd.Timedelta(minutes=5), decision + pd.Timedelta(hours=24, minutes=5)
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_ <= end), None)
        if split is None:
            continue
        rows.append({"candidate": prereg.POLICY_ID, "control": control, "split": split, "decision_time": decision, "feature_available_time": decision, "entry_time": entry, "exit_time": exit_, "side": side, "displacement": float(row.displacement), "range_log": float(row.range_log), "close_location": float(row.close_location), "range_rank": float(row.range_rank)})
    return pd.DataFrame(rows, columns=COLUMNS)


def stats(frame: pd.DataFrame, split: str) -> dict[str, Any]:
    subset = frame[frame.split.eq(split)]
    if subset.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0., "max_month_share": 0.}
    longs, shorts = int(subset.side.eq(1).sum()), int(subset.side.eq(-1).sum())
    months = pd.to_datetime(subset.entry_time, utc=True).dt.strftime("%Y-%m").value_counts()
    return {"events": len(subset), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(subset), "max_month_share": int(months.max()) / len(subset)}


def run() -> dict[str, Any]:
    bindings = {prereg.DEFAULT_OUTPUT: PREREG_SHA, HELPER: HELPER_SHA, prereg.MARKET: prereg.MARKET_SHA}
    for path, expected in bindings.items():
        if sha(path) != expected:
            raise RuntimeError(f"HVWAR binding drift: {path}")
    market, market_source = load_market()
    full, sunday = score_states(market, 48), score_states(market, 24)
    primary = build_clock(full, sunday)
    controls = {name: build_clock(full, sunday, name) for name in CONTROLS}
    STATE.parent.mkdir(parents=True, exist_ok=True); CLOCK.parent.mkdir(parents=True, exist_ok=True); CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    states = full.copy(); states["accepted"] = states.apply(_accepted, axis=1)
    _write_gzip_csv(states, STATE); _write_gzip_csv(primary, CLOCK)
    for name, frame in controls.items():
        _write_gzip_csv(frame, CONTROL_DIR / f"{name}.csv.gz")
    support = {name: stats(primary, name) for name in SPLITS}
    checks = {key: value for name, item in support.items() for key, value in ((f"{name}_minimum_events", item["events"] >= MINIMUM[name]), (f"{name}_side_balance", item["minority_side_share"] >= .2), (f"{name}_month_concentration", item["max_month_share"] <= .45))}
    passed = all(checks.values())
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    core = {"protocol_version": "hvwar_24_source_support_v1", "policy_id": prereg.POLICY_ID, "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]}, "bindings": {str(path): digest for path, digest in bindings.items()}, "source": market_source, "source_state": {"path": str(STATE), "sha256": sha(STATE), "rows": len(states)}, "completed_preentry_sources_opened": True, "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False, "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)}, "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(frame), "promotion_authorized": False} for name, frame in controls.items()}, "support": support, "support_checks": checks, "support_passed": passed, "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False, "decision": "pass_to_novelty" if passed else "terminal_source_support_reject"}
    report = {**core, "manifest_hash": prereg.canonical_hash(core)}
    RESULT.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    report = run()
    print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
