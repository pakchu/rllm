"""Source-only support gate for frozen HVSJAC-8."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_same_sign_jump_aftershock_continuation as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
START = pd.Timestamp("2023-01-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA = "9500eb575e5536a9fe2278843ebc356703a471532d7e6031ac8a2d9583602ccf"
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), END),
}
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_excitation_gate", "no_variation_gate", "jump_count_tail",
    "one_decision_stale_excitation", "direction_flip", "forced_long",
)
ROOT = Path("data/high_volatility_same_sign_jump_aftershock_continuation_sources_2023_2026")
PANEL = ROOT / "states.csv.gz"
MANIFEST = ROOT / "manifest.json"
CLOCK = Path("data/high_volatility_same_sign_jump_aftershock_continuation_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_same_sign_jump_aftershock_continuation_controls_2023_2026")
RESULT = Path("results/high_volatility_same_sign_jump_aftershock_continuation_support_2026-08-13.json")
QUERY = """SELECT ts,open,high,low,close
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


def jump_features(returns: np.ndarray) -> tuple[float, int, float, float]:
    values = np.asarray(returns, dtype=float)
    if len(values) != 95 or not np.isfinite(values).all():
        return math.nan, 0, math.nan, math.nan
    scale = float(np.median(np.abs(values)) / 0.6744897501960817)
    if not math.isfinite(scale) or scale <= 0:
        return math.nan, 0, math.nan, scale
    indices = np.flatnonzero(np.abs(values) >= 3.0 * scale)
    count = int(len(indices))
    if count < 4:
        return math.nan, count, math.nan, scale
    gaps = np.diff(indices)
    excitation = float(np.mean(gaps <= 3))
    signed_mass = float(values[indices].sum())
    if not np.isfinite([excitation, signed_mass]).all() or signed_mass == 0:
        return math.nan, count, signed_mass, scale
    return excitation, count, signed_mass, scale


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
                text(QUERY), connection, params={"start": START.to_pydatetime(), "end": END.to_pydatetime()}
            )
    finally:
        database.dispose()
    frame["ts"] = pd.to_datetime(frame.ts, utc=True)
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame.ts.duplicated().any():
        raise RuntimeError("duplicate HVSJAC source timestamps")
    frame = frame.set_index("ts").sort_index()
    rows: list[dict[str, Any]] = []
    first_decision = START.normalize() + pd.Timedelta(hours=2)
    for decision in pd.date_range(first_decision, END, freq="8h", inclusive="left"):
        index = pd.date_range(decision - pd.Timedelta(hours=8), decision, freq="1min", inclusive="left")
        window = frame.reindex(index)
        valid = bool(
            len(window) == 480 and np.isfinite(window).all().all() and window.gt(0).all().all()
            and window.high.ge(window[["open", "close"]].max(axis=1)).all()
            and window.low.le(window[["open", "close"]].min(axis=1)).all()
            and window.high.ge(window.low).all()
        )
        if valid:
            minute_open = window.open.to_numpy(float).reshape(96, 5)[:, 0]
            five_close = window.close.to_numpy(float).reshape(96, 5)[:, -1]
            returns = np.diff(np.log(five_close))
            excitation, jump_count, signed_mass, robust_scale = jump_features(returns)
            variation = float(np.sqrt(np.square(returns).sum()))
            block_return = float(np.log(five_close[-1] / minute_open[0]))
            valid = bool(
                math.isfinite(excitation) and jump_count >= 4 and math.isfinite(signed_mass)
                and signed_mass != 0 and math.isfinite(robust_scale) and robust_scale > 0
                and variation > 0 and math.isfinite(block_return)
            )
        if not valid:
            excitation = signed_mass = robust_scale = variation = block_return = math.nan
            jump_count = 0
        rows.append({
            "decision_time": decision, "source_valid": valid,
            "aftershock_excitation": excitation, "jump_count": jump_count,
            "signed_jump_mass": signed_mass, "robust_scale": robust_scale,
            "realized_variation": variation, "block_return": block_return,
        })
    states = pd.DataFrame(rows)
    states["excitation_rank"] = prior_rank(states.aftershock_excitation.where(states.source_valid))
    states["jump_count_rank"] = prior_rank(states.jump_count.where(states.source_valid))
    states["variation_rank"] = prior_rank(states.realized_variation.where(states.source_valid))
    ROOT.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(states, PANEL)
    core = {
        "protocol_version": "hvsjac_source_v1", "query": QUERY,
        "window": [START.isoformat(), END.isoformat()], "outcomes_opened": False,
        "candidate_incidence_opened_before_materialization": False,
        "output": {"path": str(PANEL), "sha256": sha256(PANEL), "rows": len(states),
                   "valid_rows": int(states.source_valid.sum())},
    }
    manifest = {**core, "manifest_hash": canonical_hash(core)}
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return states, manifest


def active(states: pd.DataFrame, control: str) -> tuple[pd.Series, pd.Series]:
    excitation_rank = states.excitation_rank
    if control == "one_decision_stale_excitation":
        excitation_rank = excitation_rank.shift(1)
    if control == "no_excitation_gate":
        excitation_gate = pd.Series(True, index=states.index)
    elif control == "jump_count_tail":
        excitation_gate = states.jump_count_rank.ge(0.75)
    else:
        excitation_gate = excitation_rank.ge(0.75)
    variation_gate = (
        pd.Series(True, index=states.index) if control == "no_variation_gate"
        else states.variation_rank.ge(0.65)
    )
    eligible = states.source_valid & excitation_gate & variation_gate & states.signed_jump_mass.ne(0)
    onset = eligible & ~eligible.shift(1, fill_value=False) & states.source_valid.shift(1, fill_value=False)
    return onset, np.sign(states.signed_jump_mass)


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
        elif control == "forced_long":
            side = 1
        reserved_until = exit_
        rows.append({
            "candidate": prereg.POLICY_ID, "control": control, "split": split,
            "decision_time": decision, "feature_available_time": decision,
            "entry_time": entry, "exit_time": exit_, "side": side,
            "aftershock_excitation": float(states.at[index, "aftershock_excitation"]),
            "excitation_rank": float(states.at[index, "excitation_rank"]),
            "jump_count": int(states.at[index, "jump_count"]),
            "signed_jump_mass": float(states.at[index, "signed_jump_mass"]),
            "realized_variation": float(states.at[index, "realized_variation"]),
            "variation_rank": float(states.at[index, "variation_rank"]),
        })
    columns = [
        "candidate", "control", "split", "decision_time", "feature_available_time", "entry_time",
        "exit_time", "side", "aftershock_excitation", "excitation_rank", "jump_count",
        "signed_jump_mass", "realized_variation", "variation_rank",
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
        raise RuntimeError("HVSJAC preregistration drift")
    states, source_manifest = materialize()
    primary = clock(states)
    controls = {name: clock(states, name) for name in CONTROLS}
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(primary, CLOCK)
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
        "protocol_version": "hvsjac_8_source_support_v1", "policy_id": prereg.POLICY_ID,
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA,
                            "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(MANIFEST), "sha256": sha256(MANIFEST),
                            "manifest_hash": source_manifest["manifest_hash"]},
        "completed_preentry_sources_opened": True,
        "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha256(CLOCK), "rows": len(primary)},
        "controls": {
            name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"),
                   "sha256": sha256(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(value),
                   "promotion_authorized": False}
            for name, value in controls.items()
        },
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    report = run()
    print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
