"""Materialize source-only HVCKIHR-8 clocks."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_cross_venue_kyle_impact_handoff_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
START = pd.Timestamp("2023-04-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
SOURCE_DIR = Path("data/high_volatility_cross_venue_kyle_impact_handoff_relay_sources_2023_2026")
PANEL = SOURCE_DIR / "eight_hour_impact_handoff_panel.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_cross_venue_kyle_impact_handoff_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_cross_venue_kyle_impact_handoff_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_cross_venue_kyle_impact_handoff_relay_support_2026-08-10.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_variation_gate",
    "no_handoff_tail",
    "perpetual_impact_dominance",
    "one_boundary_stale_handoff",
    "direction_flip",
)
COLUMNS = (
    "candidate",
    "control",
    "split",
    "decision_time",
    "feature_available_time",
    "entry_time",
    "exit_time",
    "side",
    "spot_impact", "perp_impact", "impact_handoff", "handoff_rank",
    "spot_signed_flow", "perp_signed_flow", "full_variation", "variation_rank",
)
PERP_QUERY = """
SELECT ts,open,high,low,close,volume,quote_asset_volume,taker_buy_quote
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts
"""
SPOT_QUERY = PERP_QUERY.replace("FROM bars_binance\n", "FROM bars_binance_spot\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def strict_prior_midrank(
    values: pd.Series, lookback: int = 270, minimum: int = 180
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    output = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        prior = history[-lookback:]
        if math.isfinite(current) and len(prior) >= minimum:
            array = np.asarray(prior)
            output.at[index] = (
                np.sum(array < current) + 0.5 * np.sum(array == current)
            ) / len(array)
        if math.isfinite(current):
            history.append(current)
    return output


def postgres_engine():
    from sqlalchemy import create_engine

    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(
        postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10}
    )


def _prepare(bars: pd.DataFrame) -> pd.DataFrame:
    bars=bars.copy();bars["ts"]=pd.to_datetime(bars["ts"],utc=True)
    for column in ("open", "high", "low", "close", "volume", "quote_asset_volume", "taker_buy_quote"):
        bars[column] = pd.to_numeric(bars[column], errors="coerce")
    return bars.drop_duplicates("ts",keep=False).set_index("ts").sort_index()


def _valid_window(window:pd.DataFrame)->bool:
    finite=np.isfinite(window[["open","high","low","close"]]).all(axis=1);positive=window[["open","high","low","close"]].gt(0).all(axis=1);coherent=window.high.ge(window[["open","close"]].max(axis=1))&window.low.le(window[["open","close"]].min(axis=1))&window.high.ge(window.low);volume_valid=np.isfinite(window[["volume","quote_asset_volume","taker_buy_quote"]]).all(axis=1)&window[["volume","quote_asset_volume","taker_buy_quote"]].ge(0).all(axis=1)&window.taker_buy_quote.le(window.quote_asset_volume);return len(window)==480 and bool((finite&positive&coherent&volume_valid).all())


def _venue_metrics(window: pd.DataFrame) -> tuple[float, float, np.ndarray]:
    groups = np.arange(len(window)) // 5
    five = window.groupby(groups, sort=True).agg(
        open=("open", "first"),
        close=("close", "last"),
        quote=("quote_asset_volume", "sum"),
        buy_quote=("taker_buy_quote", "sum"),
    )
    if len(five) != 96 or not five.quote.gt(0).all():
        raise ValueError("incomplete five-minute venue path")
    flow = (2.0 * five.buy_quote.to_numpy(float) - five.quote.to_numpy(float)) / five.quote.to_numpy(float)
    returns = np.log(five.close.to_numpy(float) / five.open.to_numpy(float))
    denominator = float(np.square(flow).sum())
    impact = float(np.dot(flow, returns) / denominator) if denominator > 0 else float("nan")
    signed_flow = float((2.0 * five.buy_quote - five.quote).sum())
    return impact, signed_flow, returns


def _cross_venue_kyle_impact_handoff_panel(perp: pd.DataFrame,spot:pd.DataFrame) -> pd.DataFrame:
    perp=_prepare(perp);spot=_prepare(spot)
    decisions = pd.date_range(START.ceil("8h"), END, freq="8h", inclusive="left")
    rows = []
    for decision in decisions:
        expected = pd.date_range(
            decision - pd.Timedelta(hours=8), decision, freq="1min", inclusive="left"
        )
        perp_window=perp.reindex(expected);spot_window=spot.reindex(expected);valid=_valid_window(perp_window) and _valid_window(spot_window)
        if valid:
            perp_impact, perp_signed_flow, perp_returns = _venue_metrics(perp_window)
            spot_impact, spot_signed_flow, _ = _venue_metrics(spot_window)
            full_variation = float(np.sqrt(np.square(perp_returns).sum()))
            impact_handoff = (
                float(np.log(spot_impact / perp_impact))
                if spot_impact > 0 and perp_impact > 0 else float("nan")
            )
            valid = bool(
                np.isfinite([spot_impact, perp_impact, impact_handoff, spot_signed_flow, perp_signed_flow, full_variation]).all()
                and full_variation > 0
            )
        else:
            spot_impact=perp_impact=impact_handoff=spot_signed_flow=perp_signed_flow=full_variation=float("nan")
        rows.append(
            {
                "decision_time": decision,
                "perp_source_rows":int(perp_window.notna().all(axis=1).sum()),"spot_source_rows":int(spot_window.notna().all(axis=1).sum()),
                "source_valid": valid,
                "spot_impact": spot_impact, "perp_impact": perp_impact,
                "impact_handoff": impact_handoff, "spot_signed_flow": spot_signed_flow,
                "perp_signed_flow": perp_signed_flow, "full_variation": full_variation,
            }
        )
    panel = pd.DataFrame(rows)
    panel["handoff_rank"] = strict_prior_midrank(panel.impact_handoff)
    panel["variation_rank"] = strict_prior_midrank(panel.full_variation)
    return panel


def materialize() -> dict[str, Any]:
    from sqlalchemy import text

    database = postgres_engine()
    with database.connect() as connection:
        perp = pd.read_sql_query(
            text(PERP_QUERY),
            connection,
            params={"start": START.to_pydatetime(), "end": END.to_pydatetime()},
        )
        spot=pd.read_sql_query(text(SPOT_QUERY),connection,params={"start":START.to_pydatetime(),"end":END.to_pydatetime()})
    database.dispose()
    panel = _cross_venue_kyle_impact_handoff_panel(perp,spot)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(panel, PANEL)
    core = {
        "protocol_version": "hvckihr_8_btc_source_v1",
        "queries":{"perp":PERP_QUERY,"spot":SPOT_QUERY},
        "tables": ["bars_binance","bars_binance_spot"],
        "symbol": "BTCUSDT",
        "interval": "1m",
        "window": [START.isoformat(), END.isoformat()],
        "exact_candidate_outcomes_opened": False,
        "candidate_incidence_opened": False,
        "no_imputation": True,
        "output": {
            "path": str(PANEL),
            "sha256": sha(PANEL),
            "rows": len(panel),
            "valid_rows": int(panel.source_valid.sum()),
        },
    }
    manifest = {**core, "manifest_hash": canonical_hash(core)}
    SOURCE_MANIFEST.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def features() -> pd.DataFrame:
    frame = pd.read_csv(PANEL, compression="gzip")
    frame["decision_time"] = pd.to_datetime(frame.decision_time, utc=True)
    frame["source_valid"] = frame.source_valid.astype(str).str.lower().eq("true")
    numeric = (
        "spot_impact", "perp_impact", "impact_handoff", "handoff_rank",
        "spot_signed_flow", "perp_signed_flow", "full_variation", "variation_rank",
    )
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["signal_valid"] = (
        frame.source_valid
        & np.isfinite(frame[list(numeric)]).all(axis=1)
        & frame.full_variation.gt(0)
    )
    return frame


def conditions(frame: pd.DataFrame, control: str) -> tuple[pd.Series, pd.Series]:
    spot_impact = frame.spot_impact
    perp_impact = frame.perp_impact
    handoff = frame.impact_handoff
    handoff_rank = frame.handoff_rank
    spot_flow = frame.spot_signed_flow
    perp_flow = frame.perp_signed_flow
    variation_rank = frame.variation_rank
    if control == "one_boundary_stale_handoff":
        spot_impact=spot_impact.shift(1);perp_impact=perp_impact.shift(1);handoff=handoff.shift(1)
        handoff_rank=handoff_rank.shift(1);spot_flow=spot_flow.shift(1);perp_flow=perp_flow.shift(1)
        variation_rank=variation_rank.shift(1)
    volatility_gate = (
        pd.Series(True, index=frame.index)
        if control == "no_variation_gate"
        else variation_rank.ge(0.65)
    )
    impact_gate = spot_impact.gt(0) & perp_impact.gt(0)
    if control == "perpetual_impact_dominance":
        handoff_gate = handoff.lt(0) & handoff_rank.le(0.20)
    else:
        handoff_gate = pd.Series(True, index=frame.index) if control == "no_handoff_tail" else handoff.gt(0) & handoff_rank.ge(0.80)
    long = spot_flow.gt(0) & perp_flow.gt(0)
    short = spot_flow.lt(0) & perp_flow.lt(0)
    eligible = (
        frame.signal_valid
        & np.isfinite(spot_impact) & np.isfinite(perp_impact) & np.isfinite(handoff)
        & np.isfinite(handoff_rank) & np.isfinite(spot_flow) & np.isfinite(perp_flow)
        & np.isfinite(variation_rank)
        & impact_gate
        & volatility_gate
        & handoff_gate
        & (long | short)
    )
    side = pd.Series(np.where(long, 1, -1), index=frame.index)
    if control == "direction_flip":
        side = -side
    prior_valid = frame.signal_valid.shift(1, fill_value=False)
    prior_eligible = eligible.shift(1, fill_value=False)
    return eligible & prior_valid & ~prior_eligible, side


def clock(frame: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    active, side = conditions(frame, control)
    rows = []
    for index in frame.index[active]:
        decision = pd.Timestamp(frame.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=5)
        exit_ = entry + pd.Timedelta(hours=8)
        split = next(
            (
                name
                for name, (start, end) in SPLITS.items()
                if entry >= start and exit_ <= end
            ),
            None,
        )
        if split is None:
            continue
        rows.append(
            {
                "candidate": "HVCKIHR-8",
                "control": control,
                "split": split,
                "decision_time": decision,
                "feature_available_time": decision,
                "entry_time": entry,
                "exit_time": exit_,
                "side": int(side.at[index]),
                "spot_impact":float(frame.at[index,"spot_impact"]),
                "perp_impact":float(frame.at[index,"perp_impact"]),
                "impact_handoff":float(frame.at[index,"impact_handoff"]),
                "handoff_rank":float(frame.at[index,"handoff_rank"]),
                "spot_signed_flow":float(frame.at[index,"spot_signed_flow"]),
                "perp_signed_flow":float(frame.at[index,"perp_signed_flow"]),
                "full_variation": float(frame.at[index, "full_variation"]),
                "variation_rank": float(frame.at[index, "variation_rank"]),
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def stats(candidate: pd.DataFrame, split: str) -> dict[str, float | int]:
    subset = candidate[candidate.split.eq(split)]
    if subset.empty:
        return {
            "events": 0,
            "longs": 0,
            "shorts": 0,
            "minority_side_share": 0.0,
            "max_month_share": 0.0,
        }
    longs = int(subset.side.eq(1).sum())
    shorts = int(subset.side.eq(-1).sum())
    months = subset.entry_time.dt.strftime("%Y-%m").value_counts()
    return {
        "events": len(subset),
        "longs": longs,
        "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(subset),
        "max_month_share": int(months.max()) / len(subset),
    }


def run() -> dict[str, Any]:
    source_manifest = materialize()
    frame = features()
    primary = clock(frame)
    controls = {name: clock(frame, name) for name in CONTROLS}
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(primary, CLOCK)
    for name, candidate in controls.items():
        _write_gzip_csv(candidate, CONTROL_DIR / f"{name}.csv.gz")
    support = {name: stats(primary, name) for name in SPLITS}
    checks = {}
    for name, values in support.items():
        checks[f"{name}_minimum_events"] = values["events"] >= MINIMUM[name]
        checks[f"{name}_side_balance"] = values["minority_side_share"] >= 0.2
        checks[f"{name}_month_concentration"] = values["max_month_share"] <= 0.45
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    passed = all(checks.values())
    core = {
        "protocol_version": "hvckihr_8_source_support_v1",
        "policy_id": "HVCKIHR-8",
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT),
            "sha256": sha(prereg.DEFAULT_OUTPUT),
            "manifest_hash": registration["manifest_hash"],
        },
        "source_manifest": {
            "path": str(SOURCE_MANIFEST),
            "sha256": sha(SOURCE_MANIFEST),
            "manifest_hash": source_manifest["manifest_hash"],
        },
        "completed_preentry_sources_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {
            name: {
                "path": str(CONTROL_DIR / f"{name}.csv.gz"),
                "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"),
                "rows": len(candidate),
                "promotion_authorized": False,
            }
            for name, candidate in controls.items()
        },
        "support": support,
        "support_checks": checks,
        "support_passed": passed,
        "advance_to_gross9_novelty": passed,
        "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    output = run()
    print(json.dumps({"passed": output["support_passed"], "support": output["support"]}, indent=2))
