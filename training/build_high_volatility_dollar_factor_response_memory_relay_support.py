"""Build causal source-only HVDFRM-12 clocks before Gross9 or economics."""
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

from training import preregister_high_volatility_dollar_factor_response_memory_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
from training.build_scheduled_trend_concordance_relay_support import load_market


PREREG_SHA = "1f50436d79a2fe604793ad5592dbeba348c6960b95d24f4b8b46dcd742bba64a"
HELPER = Path("training/build_scheduled_trend_concordance_relay_support.py")
HELPER_SHA = "8ca554d88506df277434f73e5eb8850426614a880110088eb91aaae3b23c154f"
END = pd.Timestamp("2026-08-01T00:00:00Z")
STATE = Path("data/high_volatility_dollar_factor_response_memory_relay_sources_2023_2026/causal_states.csv.gz")
CLOCK = Path("data/high_volatility_dollar_factor_response_memory_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_dollar_factor_response_memory_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_dollar_factor_response_memory_relay_support_2026-08-09.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), END),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_variation_gate", "fixed_inverse_factor", "fixed_direct_factor",
    "one_observation_stale_memory", "direction_flip", "same_clock_forced_long",
)
COLUMNS = (
    "candidate", "control", "split", "decision_time", "feature_available_time",
    "entry_time", "exit_time", "side", "dollar_factor", "variation",
    "variation_rank", "memory_count", "memory_mean_response",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def score_states(factors: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Create predecision scores and future labels; labels are consumed only causally."""
    f = factors.copy()
    f["decision_time"] = pd.to_datetime(f.decision_time, utc=True)
    f["source_valid"] = f.source_valid.map(_bool)
    f["dollar_factor"] = pd.to_numeric(f.dollar_factor, errors="coerce")
    f = f.sort_values("decision_time").reset_index(drop=True)
    prices = market[["date", "open"]].copy()
    prices["date"] = pd.to_datetime(prices.date, utc=True)
    prices["open"] = pd.to_numeric(prices.open, errors="coerce")
    opens = prices.set_index("date").open
    rows: list[dict[str, Any]] = []
    prior_variations: list[float] = []
    for item in f.itertuples(index=False):
        decision = pd.Timestamp(item.decision_time)
        hourly = pd.date_range(decision - pd.Timedelta(hours=24), decision, freq="1h")
        values = opens.reindex(hourly).to_numpy(dtype=float)
        variation = float(np.sqrt(np.square(np.diff(np.log(values))).sum())) if len(values) == 25 and np.isfinite(values).all() and (values > 0).all() else np.nan
        history = np.asarray(prior_variations[-90:], dtype=float)
        rank = float(((history < variation).sum() + .5 * (history == variation).sum()) / len(history)) if np.isfinite(variation) and len(history) >= 60 else np.nan
        if bool(item.source_valid) and np.isfinite(variation):
            prior_variations.append(variation)
        entry, exit_ = decision + pd.Timedelta(minutes=5), decision + pd.Timedelta(hours=12, minutes=5)
        pair = opens.reindex([entry, exit_]).to_numpy(dtype=float)
        factor = float(item.dollar_factor)
        response = float(np.sign(factor) * np.log(pair[1] / pair[0])) if np.isfinite(factor) and factor != 0 and np.isfinite(pair).all() and (pair > 0).all() else np.nan
        valid_factor = bool(item.source_valid and np.isfinite(factor) and factor != 0)
        rows.append({
            "decision_time": decision, "entry_time": entry, "exit_time": exit_,
            "source_valid": bool(item.source_valid), "dollar_factor": factor,
            "variation": variation, "variation_rank": rank,
            "valid_factor": valid_factor,
            "eligible": bool(valid_factor and np.isfinite(rank) and rank >= .65),
            "signed_response": response,
        })
    return pd.DataFrame(rows)


def causal_state(scores: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    out = scores.copy()
    out["memory_count"] = 0
    out["memory_mean_response"] = np.nan
    out["active"] = False
    out["side"] = 0
    primary_ready: dict[int, tuple[list[int], float]] = {}
    for i, row in out.iterrows():
        decision = pd.Timestamp(row.decision_time)
        eligible = out.eligible if control != "no_variation_gate" else out.valid_factor
        prior = [j for j in out.index[:i] if bool(eligible.at[j]) and pd.Timestamp(out.at[j, "exit_time"]) <= decision and np.isfinite(out.at[j, "signed_response"])]
        if control == "one_observation_stale_memory" and prior:
            prior = prior[:-1]
        used = prior[-32:]
        if control != "no_variation_gate" and not bool(row.eligible):
            continue
        if len(used) < 20:
            continue
        mean = float(out.loc[used, "signed_response"].mean())
        if not np.isfinite(mean) or mean == 0:
            continue
        primary_ready[i] = (used, mean)
        factor_side = int(np.sign(row.dollar_factor))
        if control == "fixed_inverse_factor":
            side = -factor_side
        elif control == "fixed_direct_factor":
            side = factor_side
        elif control == "same_clock_forced_long":
            side = 1
        else:
            side = factor_side * int(np.sign(mean))
            if control == "direction_flip":
                side = -side
        out.at[i, "memory_count"] = len(used)
        out.at[i, "memory_mean_response"] = mean
        out.at[i, "active"] = True
        out.at[i, "side"] = side
    return out


def clock(scores: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    state = causal_state(scores, control)
    rows = []
    for _, row in state[state.active].iterrows():
        entry, exit_ = pd.Timestamp(row.entry_time), pd.Timestamp(row.exit_time)
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_ <= end), None)
        if split is None:
            continue
        rows.append({
            "candidate": prereg.POLICY_ID, "control": control, "split": split,
            "decision_time": row.decision_time, "feature_available_time": row.decision_time,
            "entry_time": entry, "exit_time": exit_, "side": int(row.side),
            "dollar_factor": float(row.dollar_factor), "variation": float(row.variation),
            "variation_rank": float(row.variation_rank), "memory_count": int(row.memory_count),
            "memory_mean_response": float(row.memory_mean_response),
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def stats(frame: pd.DataFrame, split: str) -> dict[str, Any]:
    subset = frame[frame.split.eq(split)]
    if subset.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0., "max_month_share": 0.}
    longs, shorts = int(subset.side.eq(1).sum()), int(subset.side.eq(-1).sum())
    months = pd.to_datetime(subset.entry_time, utc=True).dt.strftime("%Y-%m").value_counts()
    return {"events": len(subset), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(subset), "max_month_share": int(months.max()) / len(subset)}


def run() -> dict[str, Any]:
    bindings = {prereg.DEFAULT_OUTPUT: PREREG_SHA, HELPER: HELPER_SHA, prereg.FACTOR: prereg.FACTOR_SHA, prereg.FACTOR_MANIFEST: prereg.FACTOR_MANIFEST_SHA, prereg.MARKET: prereg.MARKET_SHA}
    for path, expected in bindings.items():
        if sha(path) != expected:
            raise RuntimeError(f"HVDFRM binding drift: {path}")
    market, market_source = load_market()
    factors = pd.read_csv(prereg.FACTOR)
    scores = score_states(factors, market)
    primary = clock(scores)
    controls = {name: clock(scores, name) for name in CONTROLS}
    STATE.parent.mkdir(parents=True, exist_ok=True)
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    visible = causal_state(scores).drop(columns=["signed_response"])
    _write_gzip_csv(visible, STATE)
    _write_gzip_csv(primary, CLOCK)
    for name, frame in controls.items():
        _write_gzip_csv(frame, CONTROL_DIR / f"{name}.csv.gz")
    support = {name: stats(primary, name) for name in SPLITS}
    checks = {key: value for name, values in support.items() for key, value in (
        (f"{name}_minimum_events", values["events"] >= MINIMUM[name]),
        (f"{name}_side_balance", values["minority_side_share"] >= .2),
        (f"{name}_month_concentration", values["max_month_share"] <= .45),
    )}
    passed = all(checks.values())
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    core = {
        "protocol_version": "hvdfrm_12_source_support_v1", "policy_id": prereg.POLICY_ID,
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "bindings": {str(path): expected for path, expected in bindings.items()},
        "market_source": market_source,
        "causal_audit": {"causally_mature_response_label_opens_consumed": True, "same_decision_label_used": False, "unmatured_label_used": False, "funding_pnl_cagr_mdd_opened": False},
        "source_state": {"path": str(STATE), "sha256": sha(STATE), "rows": len(visible), "current_response_labels_exposed": False},
        "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(frame), "promotion_authorized": False} for name, frame in controls.items()},
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_metrics": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": chash(core)}
    RESULT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    result = run()
    print(json.dumps({"passed": result["support_passed"], "support": result["support"]}, indent=2))
