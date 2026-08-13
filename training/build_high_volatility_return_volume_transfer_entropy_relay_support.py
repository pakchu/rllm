"""Source-only support gate for frozen HVRVTE-8."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_return_volume_transfer_entropy_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

ENV_FILE = "/home/pakchu/rllm/.env"
SOURCE_START = pd.Timestamp("2022-12-31T17:00:00Z")
DECISION_START = pd.Timestamp("2023-01-01T01:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA = "6921cf0a08a7f816bf95ccf0dfdc4aea6152a161404e34f916e5ab2cbbba95c6"
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), END),
}
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_information_tail",
    "no_variation_gate",
    "unconditional_transition_lift",
    "contemporaneous_conditioned_information",
    "one_decision_stale_information",
    "direction_flip",
    "same_clock_forced_long",
)
ROOT = Path("data/high_volatility_return_volume_transfer_entropy_relay_sources_2023_2026")
PANEL = ROOT / "states.csv.gz"
MANIFEST = ROOT / "manifest.json"
CLOCK = Path("data/high_volatility_return_volume_transfer_entropy_relay_clocks_2023_2026.csv.gz")
SPLIT_DIR = Path("data/high_volatility_return_volume_transfer_entropy_relay_split_clocks_2023_2026")
CONTROL_DIR = Path("data/high_volatility_return_volume_transfer_entropy_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_return_volume_transfer_entropy_relay_support_2026-08-13.json")
QUERY = """SELECT ts,open,high,low,close,quote_asset_volume
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def prior_rank(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(float)
    output = np.full(len(values), np.nan)
    history: list[float] = []
    for index, value in enumerate(values):
        prior = np.asarray(history[-270:], dtype=float)
        if math.isfinite(value) and len(prior) >= 180:
            output[index] = (np.sum(prior < value) + 0.5 * np.sum(prior == value)) / len(prior)
        if math.isfinite(value):
            history.append(float(value))
    return pd.Series(output, index=series.index)


def conditional_information(
    turnover_state: np.ndarray,
    return_sign: np.ndarray,
    *,
    contemporaneous: bool = False,
) -> tuple[float, float, int]:
    x_all = np.asarray(turnover_state, dtype=int)
    signs = np.asarray(return_sign, dtype=int)
    if len(x_all) != 96 or len(signs) != 96:
        return math.nan, math.nan, 0
    if not np.isin(x_all, [0, 1]).all() or not np.isin(signs, [0, 1]).all():
        return math.nan, math.nan, 0
    x = x_all[1:] if contemporaneous else x_all[:-1]
    y = signs[1:]
    z = signs[:-1]
    counts = np.zeros((2, 2, 2), dtype=int)
    for xv, yv, zv in zip(x, y, z, strict=True):
        counts[xv, yv, zv] += 1
    cell_min = int(counts.sum(axis=1).min())
    if cell_min < 5:
        return math.nan, math.nan, cell_min
    total = float(len(x))
    information = 0.0
    for xv in range(2):
        for yv in range(2):
            for zv in range(2):
                n_xyz = counts[xv, yv, zv]
                if n_xyz == 0:
                    continue
                n_z = counts[:, :, zv].sum()
                n_xz = counts[xv, :, zv].sum()
                n_yz = counts[:, yv, zv].sum()
                information += (n_xyz / total) * math.log((n_xyz * n_z) / (n_xz * n_yz))
    lift = 0.0
    for zv in range(2):
        n_z = counts[:, :, zv].sum()
        high = counts[1, :, zv].sum()
        low = counts[0, :, zv].sum()
        lift += (n_z / total) * (counts[1, 1, zv] / high - counts[0, 1, zv] / low)
    if not math.isfinite(information) or not math.isfinite(lift):
        return math.nan, math.nan, cell_min
    return float(max(information, 0.0)), float(lift), cell_min


def unconditional_information(turnover_state: np.ndarray, return_sign: np.ndarray) -> tuple[float, float]:
    x = np.asarray(turnover_state, dtype=int)[:-1]
    y = np.asarray(return_sign, dtype=int)[1:]
    if len(x) != 95 or len(y) != 95 or not np.isin(x, [0, 1]).all() or not np.isin(y, [0, 1]).all():
        return math.nan, math.nan
    counts = np.zeros((2, 2), dtype=int)
    for xv, yv in zip(x, y, strict=True):
        counts[xv, yv] += 1
    if counts.sum(axis=1).min() < 5:
        return math.nan, math.nan
    total = float(len(x))
    information = 0.0
    for xv in range(2):
        for yv in range(2):
            n_xy = counts[xv, yv]
            if n_xy:
                information += (n_xy / total) * math.log(
                    (n_xy * total) / (counts[xv, :].sum() * counts[:, yv].sum())
                )
    lift = counts[1, 1] / counts[1, :].sum() - counts[0, 1] / counts[0, :].sum()
    return float(max(information, 0.0)), float(lift)


def onset_after_previous_source_valid(source_valid: pd.Series, eligible: pd.Series) -> pd.Series:
    output = pd.Series(False, index=eligible.index)
    previous_eligible = False
    for index in eligible.index:
        if not bool(source_valid.at[index]):
            continue
        current = bool(eligible.at[index])
        output.at[index] = current and not previous_eligible
        previous_eligible = current
    return output


def engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def materialize() -> tuple[pd.DataFrame, dict[str, Any]]:
    from sqlalchemy import text

    database = engine()
    try:
        with database.connect() as connection:
            frame = pd.read_sql_query(
                text(QUERY), connection, params={"start": SOURCE_START.to_pydatetime(), "end": END.to_pydatetime()}
            )
    finally:
        database.dispose()
    frame["ts"] = pd.to_datetime(frame.ts, utc=True)
    for column in ("open", "high", "low", "close", "quote_asset_volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame.ts.duplicated().any():
        raise RuntimeError("duplicate HVRVTE source timestamps")
    frame = frame.set_index("ts").sort_index()
    rows: list[dict[str, Any]] = []
    for decision in pd.date_range(DECISION_START, END, freq="8h", inclusive="left"):
        index = pd.date_range(decision - pd.Timedelta(hours=8), decision, freq="1min", inclusive="left")
        window = frame.reindex(index)
        valid = bool(
            len(window) == 480
            and np.isfinite(window).all().all()
            and window[["open", "high", "low", "close", "quote_asset_volume"]].gt(0).all().all()
            and window.high.ge(window[["open", "close"]].max(axis=1)).all()
            and window.low.le(window[["open", "close"]].min(axis=1)).all()
            and window.high.ge(window.low).all()
        )
        if valid:
            bar_open = window.open.to_numpy(float).reshape(96, 5)[:, 0]
            bar_close = window.close.to_numpy(float).reshape(96, 5)[:, -1]
            returns = np.log(bar_close / bar_open)
            turnover = window.quote_asset_volume.to_numpy(float).reshape(96, 5).sum(axis=1)
            median = float(np.median(turnover))
            turnover_state = (turnover > median).astype(int)
            return_sign = (returns > 0).astype(int)
            valid = bool(
                np.isfinite(returns).all()
                and np.all(returns != 0)
                and np.isfinite(turnover).all()
                and np.all(turnover > 0)
                and np.unique(turnover_state).size == 2
            )
        if valid:
            information, lift, cell_min = conditional_information(turnover_state, return_sign)
            contemporaneous_information, contemporaneous_lift, contemporaneous_cell_min = conditional_information(
                turnover_state, return_sign, contemporaneous=True
            )
            unconditioned_information, unconditioned_lift = unconditional_information(turnover_state, return_sign)
            variation = float(np.sqrt(np.square(returns).sum()))
            valid = bool(
                math.isfinite(information) and information > 0 and math.isfinite(lift) and lift != 0
                and math.isfinite(variation) and variation > 0
            )
        if not valid:
            information = lift = variation = math.nan
            contemporaneous_information = contemporaneous_lift = math.nan
            unconditioned_information = unconditioned_lift = math.nan
            cell_min = contemporaneous_cell_min = 0
        rows.append({
            "decision_time": decision,
            "source_valid": valid,
            "transfer_entropy": information,
            "conditional_up_lift": lift,
            "minimum_conditioning_cell_count": cell_min,
            "unconditioned_information": unconditioned_information,
            "unconditioned_up_lift": unconditioned_lift,
            "contemporaneous_information": contemporaneous_information,
            "contemporaneous_up_lift": contemporaneous_lift,
            "contemporaneous_minimum_cell_count": contemporaneous_cell_min,
            "realized_variation": variation,
        })
    states = pd.DataFrame(rows)
    valid = states.source_valid
    states["information_rank"] = prior_rank(states.transfer_entropy.where(valid))
    states["unconditioned_information_rank"] = prior_rank(states.unconditioned_information.where(valid))
    states["contemporaneous_information_rank"] = prior_rank(states.contemporaneous_information.where(valid))
    states["variation_rank"] = prior_rank(states.realized_variation.where(valid))
    ROOT.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(states, PANEL)
    core = {
        "protocol_version": "hvrvte_source_v1",
        "query": QUERY,
        "source_window": [SOURCE_START.isoformat(), END.isoformat()],
        "decision_start": DECISION_START.isoformat(),
        "outcomes_opened": False,
        "candidate_incidence_opened_before_materialization": False,
        "output": {"path": str(PANEL), "sha256": sha256(PANEL), "rows": len(states), "valid_rows": int(valid.sum())},
    }
    manifest = {**core, "manifest_hash": canonical_hash(core)}
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return states, manifest


def active(states: pd.DataFrame, control: str) -> tuple[pd.Series, pd.Series]:
    information = states.transfer_entropy
    rank = states.information_rank
    lift = states.conditional_up_lift
    if control == "unconditional_transition_lift":
        information, rank, lift = (
            states.unconditioned_information,
            states.unconditioned_information_rank,
            states.unconditioned_up_lift,
        )
    elif control == "contemporaneous_conditioned_information":
        information, rank, lift = (
            states.contemporaneous_information,
            states.contemporaneous_information_rank,
            states.contemporaneous_up_lift,
        )
    elif control == "one_decision_stale_information":
        information, rank, lift = information.shift(1), rank.shift(1), lift.shift(1)
    information_gate = pd.Series(True, index=states.index) if control == "no_information_tail" else rank.ge(0.75)
    variation_gate = pd.Series(True, index=states.index) if control == "no_variation_gate" else states.variation_rank.ge(0.65)
    eligible = states.source_valid & information.gt(0) & lift.ne(0) & information_gate & variation_gate
    return onset_after_previous_source_valid(states.source_valid, eligible), np.sign(lift)


def clock(states: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    onset, sides = active(states, control)
    rows: list[dict[str, Any]] = []
    reserved_until: pd.Timestamp | None = None
    for index in states.index[onset]:
        decision = pd.Timestamp(states.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=5)
        exit_ = entry + pd.Timedelta(hours=8)
        if reserved_until is not None and entry < reserved_until:
            continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_ <= end), None)
        if split is None:
            continue
        side = int(sides.at[index])
        if control == "direction_flip":
            side = -side
        elif control == "same_clock_forced_long":
            side = 1
        reserved_until = exit_
        rows.append({
            "candidate": prereg.POLICY_ID,
            "control": control,
            "split": split,
            "decision_time": decision,
            "feature_available_time": decision,
            "entry_time": entry,
            "exit_time": exit_,
            "side": side,
            "transfer_entropy": float(states.at[index, "transfer_entropy"]),
            "conditional_up_lift": float(states.at[index, "conditional_up_lift"]),
            "information_rank": float(states.at[index, "information_rank"]),
            "realized_variation": float(states.at[index, "realized_variation"]),
            "variation_rank": float(states.at[index, "variation_rank"]),
        })
    columns = [
        "candidate", "control", "split", "decision_time", "feature_available_time", "entry_time", "exit_time",
        "side", "transfer_entropy", "conditional_up_lift", "information_rank", "realized_variation", "variation_rank",
    ]
    return pd.DataFrame(rows, columns=columns)


def support_stats(candidate_clock: pd.DataFrame, split: str) -> dict[str, float | int]:
    selected = candidate_clock[candidate_clock.split.eq(split)]
    if selected.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(selected.side.eq(1).sum())
    shorts = int(selected.side.eq(-1).sum())
    return {
        "events": len(selected), "longs": longs, "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(selected),
        "max_month_share": int(selected.entry_time.dt.strftime("%Y-%m").value_counts().max()) / len(selected),
    }


def run() -> dict[str, Any]:
    if sha256(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVRVTE preregistration drift")
    states, source_manifest = materialize()
    primary = clock(states)
    controls = {name: clock(states, name) for name in CONTROLS}
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(primary, CLOCK)
    for split in SPLITS:
        _write_gzip_csv(primary[primary.split.eq(split)].reset_index(drop=True), SPLIT_DIR / f"{split}.csv.gz")
    for name, value in controls.items():
        _write_gzip_csv(value, CONTROL_DIR / f"{name}.csv.gz")
    support = {name: support_stats(primary, name) for name in SPLITS}
    checks: dict[str, bool] = {}
    for name, values in support.items():
        checks[f"{name}_minimum_events"] = values["events"] >= MINIMUM_EVENTS[name]
        checks[f"{name}_side_balance"] = values["minority_side_share"] >= 0.2
        checks[f"{name}_month_concentration"] = values["max_month_share"] <= 0.45
    passed = all(checks.values())
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    core = {
        "protocol_version": "hvrvte_8_source_support_v1",
        "policy_id": prereg.POLICY_ID,
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(MANIFEST), "sha256": sha256(MANIFEST), "manifest_hash": source_manifest["manifest_hash"]},
        "completed_preentry_sources_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha256(CLOCK), "rows": len(primary)},
        "split_clocks": {
            split: {"path": str(SPLIT_DIR / f"{split}.csv.gz"), "sha256": sha256(SPLIT_DIR / f"{split}.csv.gz"), "rows": int(primary.split.eq(split).sum())}
            for split in SPLITS
        },
        "controls": {
            name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha256(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(value), "promotion_authorized": False}
            for name, value in controls.items()
        },
        "support": support,
        "support_checks": checks,
        "support_passed": passed,
        "advance_to_gross9_novelty": passed,
        "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    report = run()
    print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
