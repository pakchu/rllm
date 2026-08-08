"""Build source-support clocks for FSVIBR-6 without post-entry outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import build_funding_settlement_volatility_unwind_relay_support as source
from training import preregister_funding_settlement_volatility_ignition_breakout_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


CLOCK = Path("data/funding_settlement_volatility_ignition_breakout_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/funding_settlement_volatility_ignition_breakout_relay_controls_2023_2026")
RESULT = Path("results/funding_settlement_volatility_ignition_breakout_relay_support_2026-08-08.json")
SPLITS = source.SPLITS
MINIMUM_EVENTS = source.MINIMUM_EVENTS
CONTROLS = (
    "bvol_only_expansion",
    "dvol_only_expansion",
    "no_return_tail",
    "one_settlement_stale_volatility",
    "direction_flip",
)
ECONOMIC_OUTCOMES_AUTHORIZED = False
COLUMNS = (
    "candidate",
    "control",
    "split",
    "settlement_time",
    "decision_time",
    "feature_available_time",
    "entry_time",
    "exit_time",
    "side",
    "funding_rate",
    "post_settlement_return_1h",
    "prior_abs_post_return_q60",
    "bvol_body",
    "dvol_body",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def features() -> pd.DataFrame:
    frame = source.features().copy()
    valid = (
        frame.base_valid
        & frame.funding_rate.ne(0)
        & frame.post_settlement_return_1h.ne(0)
        & np.isfinite(frame[["post_settlement_return_1h", "bvol_body", "dvol_body"]]).all(axis=1)
    )
    frame["signal_valid"] = valid
    frame["prior_abs_post_return_q60"] = (
        frame.post_settlement_return_1h.abs()
        .where(valid)
        .shift(1)
        .rolling(270, min_periods=252)
        .quantile(0.60)
    )
    return frame


def conditions(frame: pd.DataFrame, control: str) -> pd.Series:
    volatility = frame.shift(1) if control == "one_settlement_stale_volatility" else frame
    if control == "bvol_only_expansion":
        ignition = volatility.bvol_body.gt(0)
    elif control == "dvol_only_expansion":
        ignition = volatility.dvol_body.gt(0)
    else:
        ignition = volatility.bvol_body.gt(0) & volatility.dvol_body.gt(0)
    breakout = frame.post_settlement_return_1h.ne(0)
    if control != "no_return_tail":
        breakout &= frame.prior_abs_post_return_q60.notna()
        breakout &= frame.post_settlement_return_1h.abs().ge(frame.prior_abs_post_return_q60)
    stale_valid = (
        frame.signal_valid.shift(1, fill_value=False)
        if control == "one_settlement_stale_volatility"
        else pd.Series(True, index=frame.index)
    )
    return frame.signal_valid & stale_valid & ignition & breakout


def clock(frame: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    active = conditions(frame, control)
    rows: list[dict[str, Any]] = []
    next_allowed: pd.Timestamp | None = None
    for index in frame.index[active]:
        settlement = pd.Timestamp(frame.at[index, "settlement_time"])
        decision = pd.Timestamp(frame.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=6)
        if next_allowed is not None and entry < next_allowed:
            continue
        split = next(
            (name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end),
            None,
        )
        if split is None:
            continue
        side = int(np.sign(frame.at[index, "post_settlement_return_1h"]))
        if control == "direction_flip":
            side = -side
        next_allowed = exit_time
        rows.append(
            {
                "candidate": "FSVIBR-6",
                "control": control,
                "split": split,
                "settlement_time": settlement,
                "decision_time": decision,
                "feature_available_time": decision,
                "entry_time": entry,
                "exit_time": exit_time,
                "side": side,
                "funding_rate": float(frame.at[index, "funding_rate"]),
                "post_settlement_return_1h": float(frame.at[index, "post_settlement_return_1h"]),
                "prior_abs_post_return_q60": float(frame.at[index, "prior_abs_post_return_q60"]),
                "bvol_body": float(frame.at[index, "bvol_body"]),
                "dvol_body": float(frame.at[index, "dvol_body"]),
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def split_stats(clocks: pd.DataFrame, split: str) -> dict[str, int | float]:
    selected = clocks[clocks.split.eq(split)]
    if selected.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(selected.side.eq(1).sum())
    shorts = int(selected.side.eq(-1).sum())
    monthly = selected.entry_time.dt.strftime("%Y-%m").value_counts()
    return {
        "events": len(selected),
        "longs": longs,
        "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(selected),
        "max_month_share": int(monthly.max()) / len(selected),
    }


def run() -> dict[str, Any]:
    frame = features()
    primary = clock(frame)
    controls = {name: clock(frame, name) for name in CONTROLS}
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(primary, CLOCK)
    for name, control_clocks in controls.items():
        _write_gzip_csv(control_clocks, CONTROL_DIR / f"{name}.csv.gz")
    support = {name: split_stats(primary, name) for name in SPLITS}
    checks: dict[str, bool] = {}
    for name, stats in support.items():
        checks[f"{name}_minimum_events"] = stats["events"] >= MINIMUM_EVENTS[name]
        checks[f"{name}_side_balance"] = stats["minority_side_share"] >= 0.20
        checks[f"{name}_month_concentration"] = stats["max_month_share"] <= 0.45
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    volatility_manifest = source.intrahour.NONPRICE_DIR / "manifest.json"
    price_manifest = source.intrahour.PRICE_DIR / "manifest.json"
    passed = all(checks.values())
    core = {
        "protocol_version": "fsvibr_6_source_support_v1",
        "policy_id": "FSVIBR-6",
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT),
            "sha256": sha(prereg.DEFAULT_OUTPUT),
            "manifest_hash": registration["manifest_hash"],
        },
        "source_manifests": {
            "volatility": {"path": str(volatility_manifest), "sha256": sha(volatility_manifest)},
            "completed_price": {"path": str(price_manifest), "sha256": sha(price_manifest)},
            "train_funding": {"path": str(source.engine.TRAIN_FUNDING), "sha256": sha(source.engine.TRAIN_FUNDING)},
            "later_funding": {"table": "funding_rates_binance", "symbol": "BTCUSDT"},
        },
        "completed_preentry_sources_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {
            name: {
                "path": str(CONTROL_DIR / f"{name}.csv.gz"),
                "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"),
                "rows": len(control_clocks),
                "promotion_authorized": False,
            }
            for name, control_clocks in controls.items()
        },
        "support": support,
        "support_checks": checks,
        "support_passed": passed,
        "advance_to_gross9_novelty": passed,
        "advance_to_economic_outcomes": ECONOMIC_OUTCOMES_AUTHORIZED,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    output = run()
    print(json.dumps({"passed": output["support_passed"], "support": output["support"]}, indent=2))
