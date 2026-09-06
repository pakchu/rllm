"""Build source-only HVTMRF-6 clocks before Gross9 or economic outcomes."""
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

from training import preregister_high_volatility_thursday_macro_reaction_fade as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
from training.build_scheduled_trend_concordance_relay_support import load_market


PREREG_SHA = "580970980c6aed55f731e14348d20ff2aca6003c44f7c9ce8198de1bc5e81a90"
HELPER = Path("training/build_scheduled_trend_concordance_relay_support.py")
HELPER_SHA = "8ca554d88506df277434f73e5eb8850426614a880110088eb91aaae3b23c154f"
NY = "America/New_York"
END = pd.Timestamp("2026-08-01T00:00:00Z")
STATE = Path("data/high_volatility_thursday_macro_reaction_fade_sources_2023_2026/thursday_states.csv.gz")
CLOCK = Path("data/high_volatility_thursday_macro_reaction_fade_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_thursday_macro_reaction_fade_controls_2023_2026")
RESULT = Path("results/high_volatility_thursday_macro_reaction_fade_support_2026-08-09.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), END),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = ("no_variation_gate", "reaction_continuation", "half_hour_reaction_fade", "one_week_stale_reaction", "direction_flip", "same_clock_forced_long")
COLUMNS = ("candidate", "control", "split", "anchor_time", "decision_time", "feature_available_time", "entry_time", "exit_time", "side", "reaction_return", "pre_anchor_variation", "variation_rank")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def thursday_anchors(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    left = start.tz_convert(NY).normalize()
    right = end.tz_convert(NY).normalize() + pd.Timedelta(days=1)
    days = pd.date_range(left, right, freq="D", inclusive="left")
    anchors = pd.DatetimeIndex(days[days.weekday == 3] + pd.Timedelta(hours=8, minutes=30)).tz_convert("UTC")
    return anchors[(anchors >= start) & (anchors < end)]


def _valid(window: pd.DataFrame) -> bool:
    values = window[["open", "high", "low", "close"]]
    return bool(np.isfinite(values).all(axis=1).all() and values.gt(0).all(axis=1).all() and window.high.ge(window[["open", "close"]].max(axis=1)).all() and window.low.le(window[["open", "close"]].min(axis=1)).all() and window.high.ge(window.low).all())


def score_states(market: pd.DataFrame) -> pd.DataFrame:
    frame = market.copy(); frame["date"] = pd.to_datetime(frame.date, utc=True)
    frame = frame.sort_values("date").set_index("date")
    anchors = thursday_anchors(frame.index.min() + pd.Timedelta(days=2), frame.index.max() + pd.Timedelta(days=1))
    rows, prior = [], []
    for anchor in anchors:
        variation_index = pd.date_range(anchor - pd.Timedelta(hours=24, minutes=5), anchor - pd.Timedelta(minutes=5), freq="5min")
        reaction_index = pd.date_range(anchor, anchor + pd.Timedelta(hours=1), freq="5min", inclusive="left")
        variation_window, reaction = frame.reindex(variation_index), frame.reindex(reaction_index)
        if len(variation_window) != 289 or len(reaction) != 12 or not _valid(variation_window) or not _valid(reaction):
            continue
        closes = variation_window.close.to_numpy(dtype=float)
        variation = float(np.sqrt(np.square(np.diff(np.log(closes))).sum()))
        reaction_return = float(np.log(float(reaction.close.iloc[-1]) / float(reaction.open.iloc[0])))
        half_hour_return = float(np.log(float(reaction.close.iloc[5]) / float(reaction.open.iloc[0])))
        history = np.asarray(prior[-90:], dtype=float)
        rank = float(((history < variation).sum() + .5 * (history == variation).sum()) / len(history)) if len(history) >= 60 else np.nan
        rows.append({"anchor_time": anchor, "decision_time": anchor + pd.Timedelta(hours=1), "reaction_return": reaction_return, "half_hour_return": half_hour_return, "pre_anchor_variation": variation, "variation_rank": rank})
        prior.append(variation)
    return pd.DataFrame(rows)


def build_clock(states: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    rows = []
    for i, row in states.iterrows():
        reaction = float(row.reaction_return)
        decision = pd.Timestamp(row.decision_time)
        ranked = bool(np.isfinite(row.variation_rank) and row.variation_rank >= .65)
        if control == "half_hour_reaction_fade":
            reaction = float(row.half_hour_return); decision = pd.Timestamp(row.anchor_time) + pd.Timedelta(minutes=30)
        elif control == "one_week_stale_reaction":
            if i == 0:
                continue
            reaction = float(states.iloc[i - 1].reaction_return)
        if not np.isfinite(reaction) or reaction == 0 or (not ranked and control != "no_variation_gate"):
            continue
        side = -int(np.sign(reaction))
        if control in ("reaction_continuation", "direction_flip"):
            side = -side
        elif control == "same_clock_forced_long":
            side = 1
        entry, exit_ = decision + pd.Timedelta(minutes=5), decision + pd.Timedelta(hours=6, minutes=5)
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_ <= end), None)
        if split is None:
            continue
        rows.append({"candidate": prereg.POLICY_ID, "control": control, "split": split, "anchor_time": row.anchor_time, "decision_time": decision, "feature_available_time": decision, "entry_time": entry, "exit_time": exit_, "side": side, "reaction_return": reaction, "pre_anchor_variation": float(row.pre_anchor_variation), "variation_rank": float(row.variation_rank)})
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
            raise RuntimeError(f"HVTMRF binding drift: {path}")
    market, source = load_market(); states = score_states(market)
    primary = build_clock(states); controls = {name: build_clock(states, name) for name in CONTROLS}
    STATE.parent.mkdir(parents=True, exist_ok=True); CLOCK.parent.mkdir(parents=True, exist_ok=True); CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(states, STATE); _write_gzip_csv(primary, CLOCK)
    for name, frame in controls.items():
        _write_gzip_csv(frame, CONTROL_DIR / f"{name}.csv.gz")
    support = {name: stats(primary, name) for name in SPLITS}
    checks = {key: value for name, item in support.items() for key, value in ((f"{name}_minimum_events", item["events"] >= MINIMUM[name]), (f"{name}_side_balance", item["minority_side_share"] >= .2), (f"{name}_month_concentration", item["max_month_share"] <= .45))}
    passed = all(checks.values()); registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    core = {"protocol_version": "hvtmrf_6_source_support_v1", "policy_id": prereg.POLICY_ID, "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]}, "bindings": {str(path): digest for path, digest in bindings.items()}, "calendar_audit": {"timezone": NY, "weekday": "Thursday", "local_anchor": "08:30", "dst_rule": "IANA zoneinfo via pandas", "observed_release_values_opened": False}, "source": source, "source_state": {"path": str(STATE), "sha256": sha(STATE), "rows": len(states)}, "completed_preentry_sources_opened": True, "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False, "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)}, "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(frame), "promotion_authorized": False} for name, frame in controls.items()}, "support": support, "support_checks": checks, "support_passed": passed, "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False, "decision": "pass_to_novelty" if passed else "terminal_source_support_reject"}
    report = {**core, "manifest_hash": prereg.canonical_hash(core)}
    RESULT.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"); return report


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args(); report = run()
    print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
