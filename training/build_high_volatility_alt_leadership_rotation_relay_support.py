"""Outcome-blind source-support gate for frozen HVALRR-8."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import build_high_volatility_alt_breadth_diffusion_slope_relay_support as common
from training import preregister_high_volatility_alt_leadership_rotation_relay as prereg


PREREG_SHA = "a61792f4dda10bda83c64bf4afd1864a111db3660d2b0740035d570de1688ced"
START = common.START
END = common.END
REGISTRATION = prereg.build()
POLICY = REGISTRATION["policy"]
SPLITS = {name: tuple(pd.Timestamp(value) for value in bounds) for name, bounds in REGISTRATION["stages"].items()}
GATES = REGISTRATION["source_support_gates"]
CONTROLS = tuple(REGISTRATION["diagnostic_controls"]["names"])
SOURCE_DIR = Path("data/high_volatility_alt_leadership_rotation_relay_sources_2023_2026")
PANEL = SOURCE_DIR / "eight_hour_states.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_alt_leadership_rotation_relay_clocks_2023_2026.csv.gz")
SPLIT_DIR = Path("data/high_volatility_alt_leadership_rotation_relay_split_clocks_2023_2026")
CONTROL_DIR = Path("data/high_volatility_alt_leadership_rotation_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_alt_leadership_rotation_relay_support_2026-08-13.json")
BUILDER = Path(__file__).relative_to(Path.cwd())
PANEL_COLUMNS = (
    "decision_time", "feature_available_time", "source_valid", "prior_leader_index",
    "current_leader_index", "current_leader_return", "prior_leader_current_return",
    "side", "leadership_rotation", "directional_handoff", "btc_realized_variation",
    "variation_rank", "eligible",
)
CLOCK_COLUMNS = (
    "candidate", "control", "split", "decision_time", "feature_available_time", "entry_time",
    "exit_time", "side", "prior_leader_index", "current_leader_index",
    "current_leader_return", "prior_leader_current_return", "leadership_rotation",
    "directional_handoff", "btc_realized_variation", "variation_rank",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return prereg.canonical_hash(value)


def strict_prior_midrank(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    output = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        prior = np.asarray(history[-POLICY["history_decisions"] :], dtype=float)
        if math.isfinite(current) and len(prior) >= POLICY["minimum_history_decisions"]:
            output.at[index] = (np.sum(prior < current) + 0.5 * np.sum(prior == current)) / len(prior)
        if math.isfinite(current):
            history.append(float(current))
    return output


def leader_geometry(returns: np.ndarray) -> tuple[int, float, float]:
    values = np.asarray(returns, dtype=float)
    if values.shape != (6,) or not np.isfinite(values).all() or np.any(values == 0):
        return -1, math.nan, math.nan
    absolute = np.abs(values)
    total = float(absolute.sum())
    maximum = float(absolute.max())
    leaders = np.flatnonzero(absolute == maximum)
    if total <= 0 or len(leaders) != 1:
        return -1, math.nan, math.nan
    index = int(leaders[0])
    concentration = float(np.square(absolute / total).sum())
    return index, float(values[index]), concentration


def build_panel(raw: pd.DataFrame) -> pd.DataFrame:
    source = common.prepare(raw)
    rows: list[dict[str, Any]] = []
    for decision in pd.date_range(START + pd.Timedelta(hours=25), END, freq="8h", inclusive="left"):
        block_hours = pd.date_range(decision - pd.Timedelta(hours=16), decision, freq="1h", inclusive="left")
        variation_hours = pd.date_range(decision - pd.Timedelta(hours=24), decision, freq="1h", inclusive="left")
        block_index = pd.MultiIndex.from_product([block_hours, common.SYMBOLS], names=["hour_time", "symbol"])
        btc_index = pd.MultiIndex.from_product([variation_hours, ["BTCUSDT"]], names=["hour_time", "symbol"])
        block = source.reindex(block_index)
        btc = source.reindex(btc_index)
        valid = bool(block["row_valid"].eq(True).all() and btc["row_valid"].eq(True).all())
        prior_index = current_index = -1
        current_return = prior_current_return = variation = math.nan
        rotation = handoff = False
        side = 0
        if valid:
            alt = block.loc[(slice(None), list(prereg.ALTS)), "hour_return"].unstack("symbol")
            prior_returns = alt.iloc[:8].sum(axis=0).reindex(prereg.ALTS).to_numpy(float)
            current_returns = alt.iloc[8:].sum(axis=0).reindex(prereg.ALTS).to_numpy(float)
            prior_index, _, _ = leader_geometry(prior_returns)
            current_index, current_return, _ = leader_geometry(current_returns)
            if prior_index >= 0 and current_index >= 0:
                prior_current_return = float(current_returns[prior_index])
                rotation = current_index != prior_index
                handoff = bool(np.sign(current_return) == -np.sign(prior_current_return))
            side = int(np.sign(current_return)) if current_index >= 0 else 0
            variation = float(math.sqrt(btc["squared_variation"].to_numpy(float).sum()))
            valid = bool(
                prior_index >= 0 and current_index >= 0 and side != 0
                and prior_current_return != 0 and math.isfinite(prior_current_return)
                and math.isfinite(variation) and variation > 0
            )
        rows.append({
            "decision_time": decision, "feature_available_time": decision, "source_valid": valid,
            "prior_leader_index": prior_index, "current_leader_index": current_index,
            "current_leader_return": current_return,
            "prior_leader_current_return": prior_current_return, "side": side,
            "leadership_rotation": rotation, "directional_handoff": handoff,
            "btc_realized_variation": variation,
        })
    panel = pd.DataFrame(rows)
    valid = panel["source_valid"].eq(True)
    panel["variation_rank"] = strict_prior_midrank(panel["btc_realized_variation"].where(valid))
    panel["eligible"] = (
        valid & panel["leadership_rotation"] & panel["directional_handoff"]
        & panel["variation_rank"].ge(POLICY["variation_rank_min"])
    )
    return panel.loc[:, PANEL_COLUMNS]


def active(panel: pd.DataFrame, control: str = "primary"):
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    used = panel.copy()
    if control == "one_block_stale_geometry":
        columns = [
            "source_valid", "prior_leader_index", "current_leader_index", "current_leader_return",
            "prior_leader_current_return", "side", "leadership_rotation", "directional_handoff",
            "variation_rank", "feature_available_time",
        ]
        used[columns] = panel[columns].shift(1)
    valid = used["source_valid"].eq(True)
    variation = used["variation_rank"].ge(POLICY["variation_rank_min"])
    rotation = used["leadership_rotation"].eq(True)
    handoff = used["directional_handoff"].eq(True)
    state = valid & rotation & handoff & variation
    if control == "no_variation_gate": state = valid & rotation & handoff
    elif control == "identity_change_without_directional_handoff": state = valid & rotation & variation
    elif control == "current_leader_persistence": state = valid & ~rotation & variation
    side = pd.to_numeric(used["side"], errors="coerce").fillna(0).astype(int)
    if control == "direction_flip":
        side = -side
    elif control == "forced_long":
        side = side.where(side.eq(0), 1)
    return state & side.ne(0), side, used


def build_clock(panel: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    activated, side, used = active(panel, control)
    rows: list[dict[str, Any]] = []
    reserved_until = None
    for index in panel.index[activated]:
        decision = pd.Timestamp(panel.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=POLICY["entry_delay_minutes"])
        exit_time = entry + pd.Timedelta(hours=POLICY["hold_hours"])
        if reserved_until is not None and entry < reserved_until:
            continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None:
            continue
        reserved_until = exit_time
        rows.append({
            "candidate": prereg.POLICY_ID, "control": control, "split": split,
            "decision_time": decision,
            "feature_available_time": pd.Timestamp(used.at[index, "feature_available_time"]),
            "entry_time": entry, "exit_time": exit_time, "side": int(side.at[index]),
            **{column: float(used.at[index, column]) for column in CLOCK_COLUMNS[8:]},
        })
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def support_stats(clock: pd.DataFrame, split: str) -> dict[str, float | int]:
    selected = clock.loc[clock["split"].eq(split)]
    if selected.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(selected["side"].eq(1).sum())
    shorts = int(selected["side"].eq(-1).sum())
    months = pd.to_datetime(selected["entry_time"], utc=True).dt.strftime("%Y-%m").value_counts()
    return {
        "events": len(selected), "longs": longs, "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(selected),
        "max_month_share": int(months.max()) / len(selected),
    }


def run() -> dict[str, Any]:
    if sha256(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVALRR preregistration drift")
    raw = common.load_source()
    panel = build_panel(raw)
    primary = build_clock(panel)
    controls = {name: build_clock(panel, name) for name in CONTROLS}
    splits = {name: primary.loc[primary["split"].eq(name)].copy() for name in SPLITS}
    common.immutable_write(PANEL, common.gzip_csv(panel))
    common.immutable_write(CLOCK, common.gzip_csv(primary))
    for name, frame in controls.items():
        common.immutable_write(CONTROL_DIR / f"{name}.csv.gz", common.gzip_csv(frame))
    for name, frame in splits.items():
        common.immutable_write(SPLIT_DIR / f"{name}.csv.gz", common.gzip_csv(frame))
    source_core = {
        "protocol_version": "hvalrr_8_sources_v1", "query": common.SOURCE_QUERY,
        "query_sha256": hashlib.sha256(common.SOURCE_QUERY.encode()).hexdigest(),
        "tables": ["bars_binance"], "symbols": list(common.SYMBOLS),
        "window": [START.isoformat(), END.isoformat()], "physical_rows": len(raw),
        "builder": {"path": str(BUILDER), "sha256": sha256(BUILDER)},
        "panel": {"path": str(PANEL), "sha256": sha256(PANEL), "rows": len(panel), "valid_rows": int(panel["source_valid"].sum())},
        "outcomes_opened": False, "gross9_rows_opened": False, "no_imputation": True,
    }
    source_manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}
    common.immutable_write(SOURCE_MANIFEST, common.json_bytes(source_manifest))
    support = {name: support_stats(primary, name) for name in SPLITS}
    checks = {
        key: passed
        for name, stats in support.items()
        for key, passed in (
            (f"{name}_minimum_events", stats["events"] >= GATES["minimum_events"][name]),
            (f"{name}_side_balance", stats["minority_side_share"] >= GATES["minority_side_share_min"]),
            (f"{name}_month_concentration", stats["max_month_share"] <= GATES["max_month_share"]),
        )
    }
    passed = all(checks.values())
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    core = {
        "protocol_version": "hvalrr_8_source_support_v1", "policy_id": prereg.POLICY_ID,
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(SOURCE_MANIFEST), "sha256": sha256(SOURCE_MANIFEST), "manifest_hash": source_manifest["manifest_hash"]},
        "completed_preentry_sources_opened": True, "candidate_incidence_opened": True,
        "postentry_return_pnl_execution_price_opened": False, "funding_values_opened": False,
        "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha256(CLOCK), "rows": len(primary)},
        "split_artifacts": {name: {"path": str(SPLIT_DIR / f"{name}.csv.gz"), "sha256": sha256(SPLIT_DIR / f"{name}.csv.gz"), "rows": len(frame)} for name, frame in splits.items()},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha256(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(frame), "promotion_authorized": False} for name, frame in controls.items()},
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    common.immutable_write(RESULT, common.json_bytes(result))
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"passed": result["support_passed"], "support": result["support"], "result": str(RESULT)}))
