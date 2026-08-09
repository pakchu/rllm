"""Build causal source support for frozen HVDOER-12."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from training import preregister_high_volatility_daily_online_expert_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
from training.build_scheduled_trend_concordance_relay_support import load_market

PREREG_SHA = "7ae5faf4a15d91036cb56375aabf775e7c621600468bd5208a348317239ff0a1"
LOADER = Path("training/build_scheduled_trend_concordance_relay_support.py")
SOURCE_DIR = Path("data/high_volatility_daily_online_expert_relay_sources_2020_2026")
STATES = SOURCE_DIR / "causal_online_expert_states.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_daily_online_expert_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_daily_online_expert_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_daily_online_expert_relay_support_2026-08-09.json")
END = pd.Timestamp("2026-08-01T00:00:00Z")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), END),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
EXPERTS = ("momentum_6h", "reversal_6h", "momentum_24h", "reversal_24h")
CONTROLS = (
    "no_variation_gate", "fixed_momentum_6h", "fixed_momentum_24h",
    "one_day_stale_winner", "direction_flip", "same_clock_forced_long",
)
COLUMNS = (
    "candidate", "control", "split", "decision_time", "feature_available_time",
    "entry_time", "exit_time", "side", "return_6h", "return_24h",
    "winner", "winner_score", "btc_realized_variation", "variation_rank",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chash(value: Any) -> str:
    return prereg.canonical_hash(value)


def strict_prior_midrank(values: pd.Series, lookback: int = 252, minimum: int = 126) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    output = np.full(len(numeric), np.nan)
    history: list[float] = []
    for index, current in enumerate(numeric):
        prior = history[-lookback:]
        if np.isfinite(current) and len(prior) >= minimum:
            array = np.asarray(prior)
            output[index] = (
                np.count_nonzero(array < current) + 0.5 * np.count_nonzero(array == current)
            ) / len(array)
        if np.isfinite(current):
            history.append(float(current))
    return pd.Series(output, index=values.index)


def expert_sides(return_6h: float, return_24h: float) -> np.ndarray:
    return np.asarray([
        np.sign(return_6h), -np.sign(return_6h),
        np.sign(return_24h), -np.sign(return_24h),
    ], dtype=int)


def online_states(market: pd.DataFrame, memory: int = 60, minimum: int = 30) -> pd.DataFrame:
    frame = market.copy().sort_values("date").drop_duplicates("date", keep="last")
    frame["date"] = pd.to_datetime(frame.date, utc=True)
    frame = frame.set_index("date")
    for column in ("open", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    first = frame.index.min().ceil("D") + pd.Timedelta(hours=3)
    decisions = pd.date_range(first, END, freq="1D", inclusive="left")
    histories: list[list[float]] = [[] for _ in EXPERTS]
    rows: list[dict[str, Any]] = []
    for decision in decisions:
        close_times = pd.date_range(
            decision - pd.Timedelta(hours=24, minutes=5),
            decision - pd.Timedelta(minutes=5), freq="5min",
        )
        entry, exit_time = decision + pd.Timedelta(minutes=5), decision + pd.Timedelta(hours=12, minutes=5)
        required = close_times.union(pd.DatetimeIndex([entry, exit_time]))
        if not required.isin(frame.index).all():
            continue
        closes = frame.loc[close_times, "close"].to_numpy(float)
        opens = frame.loc[[entry, exit_time], "open"].to_numpy(float)
        valid = bool(
            len(closes) == 289 and np.isfinite(closes).all() and np.all(closes > 0)
            and np.isfinite(opens).all() and np.all(opens > 0)
        )
        if not valid:
            continue
        return_6h = float(np.log(closes[-1] / closes[-73]))
        return_24h = float(np.log(closes[-1] / closes[0]))
        variation = float(np.sqrt(np.square(np.diff(np.log(closes))).sum()))
        sides = expert_sides(return_6h, return_24h)
        scores = np.asarray([
            float(np.mean(history[-memory:])) if len(history) >= minimum else np.nan
            for history in histories
        ])
        winner_index = int(np.nanargmax(scores)) if np.isfinite(scores).any() else -1
        rows.append({
            "decision_time": decision, "source_valid": bool(return_6h != 0 and return_24h != 0),
            "return_6h": return_6h, "return_24h": return_24h,
            "btc_realized_variation": variation,
            **{f"{name}_side": int(sides[index]) for index, name in enumerate(EXPERTS)},
            **{f"{name}_score": float(scores[index]) for index, name in enumerate(EXPERTS)},
            "winner": EXPERTS[winner_index] if winner_index >= 0 else "",
            "winner_side": int(sides[winner_index]) if winner_index >= 0 else 0,
            "winner_score": float(scores[winner_index]) if winner_index >= 0 else np.nan,
            "mature_labels_before_decision": min((len(history) for history in histories), default=0),
        })
        # This label is read only after the current decision has been frozen. Its exit is
        # more than eleven hours before the next daily decision, so it is mature then.
        forward = float(np.log(opens[1] / opens[0]))
        for index, history in enumerate(histories):
            history.append(float(sides[index] * forward))
    states = pd.DataFrame(rows)
    states["variation_rank"] = strict_prior_midrank(states.btc_realized_variation.where(states.source_valid))
    states["signal_valid"] = (
        states.source_valid & states.winner.ne("") & states.winner_side.ne(0)
        & np.isfinite(states[["winner_score", "btc_realized_variation", "variation_rank"]]).all(axis=1)
    )
    return states


def conditions(states: pd.DataFrame, control: str = "primary") -> tuple[pd.Series, pd.Series]:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    active = states.signal_valid.copy()
    side = states.winner_side.astype(int).copy()
    if control == "fixed_momentum_6h":
        side = states.momentum_6h_side.astype(int)
    elif control == "fixed_momentum_24h":
        side = states.momentum_24h_side.astype(int)
    elif control == "one_day_stale_winner":
        side = states.winner_side.shift(1).fillna(0).astype(int)
        active &= side.ne(0)
    elif control == "direction_flip":
        side = -side
    elif control == "same_clock_forced_long":
        side = pd.Series(1, index=states.index)
    if control != "no_variation_gate":
        active &= states.variation_rank.ge(0.65)
    return active, side


def clock(states: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    active, sides = conditions(states, control)
    rows = []
    next_allowed = None
    for index in states.index[active]:
        decision = pd.Timestamp(states.at[index, "decision_time"])
        entry, exit_time = decision + pd.Timedelta(minutes=5), decision + pd.Timedelta(hours=12, minutes=5)
        if next_allowed is not None and entry < next_allowed:
            continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None:
            continue
        next_allowed = exit_time
        rows.append({
            "candidate": "HVDOER-12", "control": control, "split": split,
            "decision_time": decision, "feature_available_time": decision,
            "entry_time": entry, "exit_time": exit_time, "side": int(sides.at[index]),
            "return_6h": float(states.at[index, "return_6h"]),
            "return_24h": float(states.at[index, "return_24h"]),
            "winner": states.at[index, "winner"], "winner_score": float(states.at[index, "winner_score"]),
            "btc_realized_variation": float(states.at[index, "btc_realized_variation"]),
            "variation_rank": float(states.at[index, "variation_rank"]),
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def stats(candidate: pd.DataFrame, split: str) -> dict[str, int | float]:
    selected = candidate[candidate.split.eq(split)]
    if selected.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs, shorts = int(selected.side.eq(1).sum()), int(selected.side.eq(-1).sum())
    months = pd.to_datetime(selected.entry_time, utc=True).dt.strftime("%Y-%m").value_counts()
    return {"events": len(selected), "longs": longs, "shorts": shorts,
            "minority_side_share": min(longs, shorts) / len(selected),
            "max_month_share": int(months.max()) / len(selected)}


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA or sha(prereg.MARKET) != prereg.MARKET_SHA:
        raise RuntimeError("HVDOER preregistration or historical source drift")
    market, source = load_market()
    states = online_states(market)
    primary = clock(states)
    controls = {name: clock(states, name) for name in CONTROLS}
    SOURCE_DIR.mkdir(parents=True, exist_ok=True); CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(states, STATES); _write_gzip_csv(primary, CLOCK)
    for name, candidate in controls.items():
        _write_gzip_csv(candidate, CONTROL_DIR / f"{name}.csv.gz")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    source_core = {
        "protocol_version": "hvdoer_12_causal_source_v1", "policy_id": "HVDOER-12",
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "market": source, "loader": {"path": str(LOADER), "sha256": sha(LOADER)},
        "states": {"path": str(STATES), "sha256": sha(STATES), "rows": len(states)},
        "causally_mature_counterfactual_labels_opened": True,
        "same_decision_label_used": False, "unmatured_label_used": False,
        "funding_pnl_cagr_mdd_opened": False, "gross9_rows_opened": False,
    }
    source_manifest = {**source_core, "manifest_hash": chash(source_core)}
    SOURCE_MANIFEST.write_text(json.dumps(source_manifest, indent=2, allow_nan=False) + "\n")
    support = {name: stats(primary, name) for name in SPLITS}; checks = {}
    for name, values in support.items():
        checks[f"{name}_minimum_events"] = values["events"] >= MINIMUM[name]
        checks[f"{name}_side_balance"] = values["minority_side_share"] >= 0.2
        checks[f"{name}_month_concentration"] = values["max_month_share"] <= 0.45
    passed = all(checks.values())
    core = {
        "protocol_version": "hvdoer_12_source_support_v1", "policy_id": "HVDOER-12",
        "preregistration": source_core["preregistration"],
        "source_manifest": {"path": str(SOURCE_MANIFEST), "sha256": sha(SOURCE_MANIFEST), "manifest_hash": source_manifest["manifest_hash"]},
        "causal_label_audit": {"mature_counterfactual_labels_opened": True, "same_decision_label_used": False, "unmatured_label_used": False},
        "current_decision_future_used": False, "funding_pnl_cagr_mdd_opened": False, "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(candidate), "promotion_authorized": False} for name, candidate in controls.items()},
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": chash(core)}
    RESULT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args(); result = run()
    print(json.dumps({"passed": result["support_passed"], "support": result["support"]}, indent=2))
