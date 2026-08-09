"""Build source-only support for frozen HVBDRR-8."""
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

from training import preregister_high_volatility_btc_dominance_rotation_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

PREREG_SHA = "307a9c751cc93709421f1c15f3f6e5e6d42508eaafc60cebaabfff22f93a3440"
SOURCE = Path(
    "data/cross_alt_breadth_underreaction_relay_sources_2023_2026/"
    "cross_alt_breadth_underreaction_relay_preentry_features.csv.gz"
)
SOURCE_SHA = "8270b0318d11d16b6b384e64bfaac77ef4bbc4a701dd347c2ababbb093061eae"
SOURCE_MANIFEST = SOURCE.parent / "manifest.json"
SOURCE_MANIFEST_SHA = "8764b8c24ce32e36fbacbfd2909c1d500efdbfbdd6b07fa6d4eb31d827c65eaa"
CLOCK = Path("data/high_volatility_btc_dominance_rotation_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_btc_dominance_rotation_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_btc_dominance_rotation_relay_support_2026-08-09.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_residual_tail",
    "no_dispersion_gate",
    "no_variation_gate",
    "alt_factor_direction",
    "one_block_stale_geometry",
    "direction_flip",
    "same_clock_forced_long",
)
ALT_RETURN_COLUMNS = tuple(f"{symbol.lower()}_return" for symbol in prereg.ALTS)
COLUMNS = (
    "candidate", "control", "split", "session_date", "decision_time",
    "feature_available_time", "entry_time", "exit_time", "side", "btc_return",
    "alt_factor", "btc_dominance_residual", "absolute_residual_rank", "alt_dispersion",
    "alt_dispersion_rank", "btc_realized_variation", "variation_rank",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def strict_prior_midrank(
    values: pd.Series, lookback: int = 270, minimum: int = 180
) -> pd.Series:
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


def features() -> pd.DataFrame:
    if sha(SOURCE) != SOURCE_SHA or sha(SOURCE_MANIFEST) != SOURCE_MANIFEST_SHA:
        raise RuntimeError("HVBDRR source drift")
    frame = pd.read_csv(SOURCE, compression="gzip")
    frame["decision_time"] = pd.to_datetime(frame.decision_time, utc=True)
    numeric = ["btc_return", *ALT_RETURN_COLUMNS, "btc_realized_variation", "variation_rank"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    alt_returns = frame[list(ALT_RETURN_COLUMNS)].to_numpy(float)
    frame["alt_factor"] = np.median(alt_returns, axis=1)
    frame["btc_dominance_residual"] = frame.btc_return - frame.alt_factor
    frame["alt_dispersion"] = np.median(
        np.abs(alt_returns - frame.alt_factor.to_numpy(float)[:, None]), axis=1
    )
    valid = np.isfinite(
        frame[["btc_return", *ALT_RETURN_COLUMNS, "btc_realized_variation", "variation_rank"]]
    ).all(axis=1)
    frame["source_valid"] = valid & frame.btc_dominance_residual.ne(0) & frame.alt_dispersion.gt(0)
    frame["absolute_residual_rank"] = strict_prior_midrank(
        frame.btc_dominance_residual.abs().where(frame.source_valid)
    )
    frame["alt_dispersion_rank"] = strict_prior_midrank(
        frame.alt_dispersion.where(frame.source_valid)
    )
    return frame


def conditions(frame: pd.DataFrame, control: str = "primary") -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    used = frame.shift(1) if control == "one_block_stale_geometry" else frame
    residual_gate = (
        pd.Series(True, index=frame.index)
        if control == "no_residual_tail"
        else used.absolute_residual_rank.ge(0.70)
    )
    dispersion_gate = (
        pd.Series(True, index=frame.index)
        if control == "no_dispersion_gate"
        else used.alt_dispersion_rank.ge(0.65)
    )
    variation_source = frame if control == "one_block_stale_geometry" else used
    variation_gate = (
        pd.Series(True, index=frame.index)
        if control == "no_variation_gate"
        else variation_source.variation_rank.ge(0.65)
    )
    active = used.source_valid.fillna(False).astype(bool) & residual_gate & dispersion_gate & variation_gate
    side = np.sign(used.btc_dominance_residual).fillna(0).astype(int)
    if control == "alt_factor_direction":
        side = np.sign(used.alt_factor).fillna(0).astype(int)
        active &= side.ne(0)
    elif control == "direction_flip":
        side = -side
    elif control == "same_clock_forced_long":
        side = pd.Series(1, index=frame.index)
    return active, side, used


def clock(frame: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    active, sides, used = conditions(frame, control)
    rows = []
    next_allowed = None
    for index in frame.index[active]:
        decision = pd.Timestamp(frame.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=8)
        if next_allowed is not None and entry < next_allowed:
            continue
        split = next(
            (name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end),
            None,
        )
        if split is None:
            continue
        source = used.loc[index]
        next_allowed = exit_time
        rows.append(
            {
                "candidate": "HVBDRR-8", "control": control, "split": split,
                "session_date": source.session_date, "decision_time": decision,
                "feature_available_time": decision, "entry_time": entry, "exit_time": exit_time,
                "side": int(sides.at[index]), "btc_return": float(source.btc_return),
                "alt_factor": float(source.alt_factor),
                "btc_dominance_residual": float(source.btc_dominance_residual),
                "absolute_residual_rank": float(source.absolute_residual_rank),
                "alt_dispersion": float(source.alt_dispersion),
                "alt_dispersion_rank": float(source.alt_dispersion_rank),
                "btc_realized_variation": float(source.btc_realized_variation),
                "variation_rank": float(variation_source_value(frame, index, control)),
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def variation_source_value(frame: pd.DataFrame, index: int, control: str) -> float:
    source_index = index if control == "one_block_stale_geometry" else index
    return frame.at[source_index, "variation_rank"]


def stats(candidate: pd.DataFrame, split: str) -> dict[str, int | float]:
    selected = candidate[candidate.split.eq(split)]
    if selected.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(selected.side.eq(1).sum())
    shorts = int(selected.side.eq(-1).sum())
    months = pd.to_datetime(selected.entry_time, utc=True).dt.strftime("%Y-%m").value_counts()
    return {
        "events": len(selected), "longs": longs, "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(selected),
        "max_month_share": int(months.max()) / len(selected),
    }


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVBDRR preregistration hash drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    frame = features()
    primary = clock(frame)
    controls = {name: clock(frame, name) for name in CONTROLS}
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(primary, CLOCK)
    for name, candidate in controls.items():
        _write_gzip_csv(candidate, CONTROL_DIR / f"{name}.csv.gz")
    support = {name: stats(primary, name) for name in SPLITS}
    checks = {}
    for name, values in support.items():
        checks[f"{name}_minimum_events"] = values["events"] >= MINIMUM[name]
        checks[f"{name}_side_balance"] = values["minority_side_share"] >= 0.20
        checks[f"{name}_month_concentration"] = values["max_month_share"] <= 0.45
    passed = all(checks.values())
    source_manifest = json.loads(SOURCE_MANIFEST.read_text())
    core = {
        "protocol_version": "hvbdrr_8_source_support_v1", "policy_id": "HVBDRR-8",
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(SOURCE_MANIFEST), "sha256": SOURCE_MANIFEST_SHA, "manifest_hash": source_manifest["manifest_hash"]},
        "source_features": {"path": str(SOURCE), "sha256": SOURCE_SHA, "rows": len(frame)},
        "completed_preentry_sources_opened": True,
        "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False,
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
    argparse.ArgumentParser().parse_args()
    result = run()
    print(json.dumps({"passed": result["support_passed"], "support": result["support"]}, indent=2))
