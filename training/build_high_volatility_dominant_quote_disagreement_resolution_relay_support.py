"""Build source-only HVDQDR-8 clocks before Gross9 or economics."""
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

from training import preregister_high_volatility_dominant_quote_disagreement_resolution_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
from training.build_scheduled_trend_concordance_relay_support import load_market


PREREG_SHA = "74bc5c30eb1243e4f5a46b253f64c70fffe17e72a5178aa133809a49c654db3c"
HELPER = Path("training/build_scheduled_trend_concordance_relay_support.py")
HELPER_SHA = "8ca554d88506df277434f73e5eb8850426614a880110088eb91aaae3b23c154f"
END = pd.Timestamp("2026-08-01T00:00:00Z")
STATE = Path("data/high_volatility_dominant_quote_disagreement_resolution_relay_sources_2023_2026/block_states.csv.gz")
CLOCK = Path("data/high_volatility_dominant_quote_disagreement_resolution_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_dominant_quote_disagreement_resolution_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_dominant_quote_disagreement_resolution_relay_support_2026-08-09.json")
SYMBOLS = ("BTCUSDT", "BTCUSDC", "BTCFDUSD")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), END),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = ("no_variation_gate", "no_activity_gate", "alternative_direction", "one_block_stale_disagreement", "direction_flip", "same_clock_forced_long")
COLUMNS = ("candidate", "control", "split", "decision_time", "feature_available_time", "entry_time", "exit_time", "side", "intensity_usdt", "intensity_usdc", "intensity_fdusd", "activity_threshold_usdt", "activity_threshold_usdc", "activity_threshold_fdusd", "btc_variation", "variation_rank")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def block_panel(flow: pd.DataFrame) -> pd.DataFrame:
    x = flow.copy()
    x["date"] = pd.to_datetime(x.date, utc=True, errors="raise")
    x = x[x.symbol.isin(SYMBOLS)].copy()
    if x[["date", "symbol"]].duplicated().any():
        raise RuntimeError("HVDQDR duplicate quote hour")
    for column in ("base_volume_btc", "trade_count", "signed_taker_flow_btc"):
        x[column] = pd.to_numeric(x[column], errors="coerce")
    x["source_complete"] = x.source_complete.astype(str).str.lower().eq("true")
    x["block_start"] = x.date.dt.floor("8h")
    grouped = x.groupby(["block_start", "symbol"], as_index=False).agg(
        hours=("date", "size"), first=("date", "min"), last=("date", "max"),
        complete=("source_complete", "all"), volume=("base_volume_btc", "sum"),
        nonnegative_volume=("base_volume_btc", lambda values: bool(values.ge(0).all())),
        trade_count=("trade_count", "sum"), flow=("signed_taker_flow_btc", "sum"),
    )
    grouped["valid"] = (
        grouped.hours.eq(8)
        & grouped["first"].eq(grouped.block_start)
        & grouped["last"].eq(grouped.block_start + pd.Timedelta(hours=7))
        & grouped.complete
        & grouped.nonnegative_volume
        & np.isfinite(grouped[["volume", "flow"]]).all(axis=1)
        & grouped.volume.gt(0)
    )
    grouped["intensity"] = (grouped.flow / grouped.volume).where(grouped.valid)
    wide = grouped.pivot(index="block_start", columns="symbol", values=["valid", "intensity"])
    wide.columns = [f"{kind}_{symbol}" for kind, symbol in wide.columns]
    wide = wide.reset_index().sort_values("block_start").reset_index(drop=True)
    for symbol in SYMBOLS:
        for kind in ("valid", "intensity"):
            column = f"{kind}_{symbol}"
            if column not in wide:
                wide[column] = np.nan
        wide[f"intensity_{symbol}"] = pd.to_numeric(wide[f"intensity_{symbol}"], errors="coerce")
    wide["block_valid"] = wide[[f"valid_{symbol}" for symbol in SYMBOLS]].eq(True).all(axis=1)
    wide["decision_time"] = wide.block_start + pd.Timedelta(hours=8)
    return wide


def _market_variation(market: pd.DataFrame, decision: pd.Timestamp) -> float:
    frame = market.set_index("date") if "date" in market else market
    expected = pd.date_range(decision - pd.Timedelta(hours=24, minutes=5), decision - pd.Timedelta(minutes=5), freq="5min")
    window = frame.reindex(expected)
    close = pd.to_numeric(window.close, errors="coerce").to_numpy(float)
    if len(close) != 289 or not np.isfinite(close).all() or not (close > 0).all():
        return np.nan
    return float(np.sqrt(np.square(np.diff(np.log(close))).sum()))


def score_states(flow: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    states = block_panel(flow)
    market = market.copy()
    market["date"] = pd.to_datetime(market.date, utc=True)
    market = market.sort_values("date").set_index("date")
    states["btc_variation"] = [_market_variation(market, pd.Timestamp(t)) for t in states.decision_time]
    ranks, history = [], []
    for value, block_valid in zip(states.btc_variation, states.block_valid):
        prior = np.asarray(history[-270:], dtype=float)
        valid = bool(block_valid and np.isfinite(value))
        ranks.append(float(((prior < value).sum() + 0.5 * (prior == value).sum()) / len(prior)) if valid and len(prior) >= 180 else np.nan)
        if valid:
            history.append(float(value))
    states["variation_rank"] = ranks
    for symbol in SYMBOLS:
        values = states[f"intensity_{symbol}"].abs().where(states.block_valid)
        states[f"activity_threshold_{symbol}"] = values.rolling(180, min_periods=90).median().shift(1)
    usdt, usdc, fdusd = states.intensity_BTCUSDT, states.intensity_BTCUSDC, states.intensity_BTCFDUSD
    states["alternative_consensus"] = states.block_valid & usdc.ne(0) & fdusd.ne(0) & np.sign(usdc).eq(np.sign(fdusd))
    states["disagreement"] = states.alternative_consensus & usdt.ne(0) & np.sign(usdt).eq(-np.sign(usdc))
    states["activity_gate"] = states.block_valid
    for symbol in SYMBOLS:
        states["activity_gate"] &= states[f"intensity_{symbol}"].abs().ge(states[f"activity_threshold_{symbol}"])
    return states


def build_clock(states: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    high_variation = states.variation_rank.ge(0.65)
    disagreement = states.disagreement
    if control == "one_block_stale_disagreement":
        disagreement = states.disagreement.shift(1).eq(True) & states.decision_time.diff().eq(pd.Timedelta(hours=8))
    active = disagreement & (True if control == "no_activity_gate" else states.activity_gate) & (True if control == "no_variation_gate" else high_variation)
    rows: list[dict[str, Any]] = []
    reserved_until: pd.Timestamp | None = None
    for i in states.index[active]:
        decision = pd.Timestamp(states.at[i, "decision_time"])
        entry, exit_ = decision + pd.Timedelta(minutes=5), decision + pd.Timedelta(hours=8, minutes=5)
        if reserved_until is not None and entry < reserved_until:
            continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_ <= end), None)
        if split is None:
            continue
        side = int(np.sign(float(states.at[i, "intensity_BTCUSDT"])))
        if control == "alternative_direction":
            side = int(np.sign(float(states.at[i, "intensity_BTCUSDC"])))
        if control == "direction_flip":
            side = -side
        elif control == "same_clock_forced_long":
            side = 1
        reserved_until = exit_
        rows.append({
            "candidate": prereg.POLICY_ID, "control": control, "split": split,
            "decision_time": decision, "feature_available_time": decision,
            "entry_time": entry, "exit_time": exit_, "side": side,
            "intensity_usdt": float(states.at[i, "intensity_BTCUSDT"]), "intensity_usdc": float(states.at[i, "intensity_BTCUSDC"]), "intensity_fdusd": float(states.at[i, "intensity_BTCFDUSD"]),
            "activity_threshold_usdt": float(states.at[i, "activity_threshold_BTCUSDT"]), "activity_threshold_usdc": float(states.at[i, "activity_threshold_BTCUSDC"]), "activity_threshold_fdusd": float(states.at[i, "activity_threshold_BTCFDUSD"]),
            "btc_variation": float(states.at[i, "btc_variation"]), "variation_rank": float(states.at[i, "variation_rank"]),
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, float | int]:
    subset = clock[clock.split.eq(split)]
    if subset.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs, shorts = int(subset.side.eq(1).sum()), int(subset.side.eq(-1).sum())
    months = pd.to_datetime(subset.entry_time, utc=True).dt.strftime("%Y-%m").value_counts()
    return {"events": len(subset), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(subset), "max_month_share": int(months.max()) / len(subset)}


def run() -> dict[str, Any]:
    bindings = {prereg.DEFAULT_OUTPUT: PREREG_SHA, HELPER: HELPER_SHA, prereg.FLOW: prereg.FLOW_SHA, prereg.FLOW_MANIFEST: prereg.FLOW_MANIFEST_SHA, prereg.MARKET: prereg.MARKET_SHA}
    for path, expected in bindings.items():
        if sha(path) != expected:
            raise RuntimeError(f"HVDQDR binding drift: {path}")
    flow = pd.read_csv(prereg.FLOW, compression="gzip")
    market, market_source = load_market()
    states = score_states(flow, market)
    primary = build_clock(states)
    controls = {name: build_clock(states, name) for name in CONTROLS}
    STATE.parent.mkdir(parents=True, exist_ok=True); CLOCK.parent.mkdir(parents=True, exist_ok=True); CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(states, STATE); _write_gzip_csv(primary, CLOCK)
    for name, frame in controls.items():
        _write_gzip_csv(frame, CONTROL_DIR / f"{name}.csv.gz")
    support = {name: stats(primary, name) for name in SPLITS}
    checks = {key: value for name, item in support.items() for key, value in ((f"{name}_minimum_events", item["events"] >= MINIMUM[name]), (f"{name}_side_balance", item["minority_side_share"] >= 0.2), (f"{name}_month_concentration", item["max_month_share"] <= 0.45))}
    passed = all(checks.values())
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    core = {
        "protocol_version": "hvdqdr_8_source_support_v1", "policy_id": prereg.POLICY_ID,
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "bindings": {str(path): expected for path, expected in bindings.items()}, "market_source": market_source,
        "information_embargo_audit": {"completed_preentry_sources_opened": True, "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False},
        "source_state": {"path": str(STATE), "sha256": sha(STATE), "rows": len(states)},
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(frame), "promotion_authorized": False} for name, frame in controls.items()},
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": prereg.canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    outcome = run()
    print(json.dumps({"passed": outcome["support_passed"], "support": outcome["support"]}, indent=2))
